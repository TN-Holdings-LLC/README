
// chaos_injector_imu.cpp -- Corrected Version

//

// This environment does not have ROS2/rclcpp installed, so the full node

// could not be built and run end-to-end here. Instead, the two numeric

// bugs below were isolated into small, ROS2-independent standalone C++

// programs (using plain doubles for "time" instead of rclcpp::Time) that

// reproduce the pasted formulas exactly and were actually compiled and

// run -- see test_chaos_injector_imu_logic.cpp for both, with the

// concrete output numbers repeated in the comments below. Everything

// ROS2-specific (topics, QoS, the timer, subscription/publisher wiring)

// is otherwise unchanged from the pasted file.

//

// 1. CLOCK SKEW WAS ACCUMULATED PER-MESSAGE, NOT PER ELAPSED TIME -- AND

//    THEN LEAKED INTO THE REAL PUBLISH SCHEDULE.

//    The pasted code:

//        static double skew_acc = 0.0;

//        skew_acc += (prm_.skew_ppm * 1e-6) * d_ms;

//        double total_ms = d_ms + skew_acc;

//        ...

//        double hold_ms = std::max(0.0, total_ms);

//        rclcpp::Time due = now() + Duration::from_nanoseconds(hold_ms * 1e6);

//    Two separate problems here:

//    (a) `skew_acc` accumulates skew_ppm*1e-6*d_ms once per MESSAGE, using

//        the randomly generated per-message network delay `d_ms` as the

//        "elapsed time" proxy. A real clock's drift depends on elapsed

//        WALL-CLOCK time, not on how many messages happened to arrive or

//        how large their simulated delays were. Verified by running the

//        exact pasted formula at different IMU publish rates for the

//        same simulated 1-hour span: skew_acc came out to 19.6 ms at

//        10 Hz, 98.4 ms at 50 Hz, 197.8 ms at 100 Hz, and 397.5 ms at

//        200 Hz -- a 20x difference for the identical elapsed real time,

//        purely because of how fast the topic happens to publish. A

//        100 ppm crystal doesn't drift 20x faster just because more ROS

//        messages flow past it. It also never resets or saturates: by

//        24 simulated hours at 200 Hz it reached 9.5 REAL SECONDS.

//    (b) That same inflated `total_ms` (which includes the runaway

//        skew_acc) was used not only for the message's reported

//        timestamp but ALSO for `hold_ms`, i.e. the ACTUAL wall-clock

//        delay before the message is really published. Real clock skew

//        should only make a sensor's reported timestamp wrong -- it

//        should never make the packet physically arrive later. As

//        written, by hour 24 of a long-running test, every single IMU

//        message would sit in the delay buffer for several EXTRA

//        SECONDS on top of the intended +-20 ms delay/jitter window,

//        because the (already wrong) skew value was leaking into real

//        scheduling.

//    Fixed by (i) computing skew from actual elapsed wall-clock time

//    since the node started (via a stored start_time_, using the node's

//    own clock -- rate-independent by construction) instead of an

//    accumulator driven by random per-message delays, and (ii) applying

//    skew ONLY to the reported header.stamp, never to the real

//    scheduling delay (`hold_ms`/`due`), which now uses `d_ms` alone.

//    skew_acc is also promoted from a function-local `static double`

//    (silently shared across every instance of this node in the same

//    process) to a proper per-instance member.

//

// 2. THE PACKET-REORDER LOGIC SWAPPED WHOLE BUFFER ENTRIES, WHICH DOES

//    NOTHING TO THE ACTUAL DELIVERY ORDER.

//    The pasted code did `std::swap(buf_[i], buf_[j])` -- swapping the

//    ENTIRE ImuItem struct, `due` timestamp included. Since drain()

//    decides what to publish purely by comparing each item's OWN `due`

//    to `now` (never by its position in the deque), moving a full item

//    (with its due time still attached) to a different slot changes

//    nothing about when it will actually be published. Verified with a

//    standalone test: swapping two full entries left the eventual

//    publish order (sorted by due) byte-for-byte identical to the

//    unswapped case, for every pair tried. Fixed by swapping only the

//    `due` timestamps between the two selected slots (leaving payload

//    and orig_stamp in place), so the two messages actually trade

//    delivery times with each other -- confirmed with the same test to

//    genuinely change the resulting publish order.

//

// Everything else in this file (topic/QoS wiring, drop/dup/burst

// handling, the delay-buffer drain loop's due<=now scan) is unchanged

// from the pasted version.

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/imu.hpp"

#include "psf_zero_eit/chaos_utils.hpp"

#include <deque>

using sensor_msgs::msg::Imu;

struct ImuItem { Imu::SharedPtr msg; rclcpp::Time due; rclcpp::Time orig_stamp; };

class ChaosInjectorImu : public rclcpp::Node {

public:

  ChaosInjectorImu() : Node("chaos_injector_imu") {

    declare_parameter<std::string>("in_topic",  "/imu/raw");

    declare_parameter<std::string>("out_topic", "/chaos/imu/raw");

    prm_.delay_min_ms   = declare_parameter<double>("delay_min_ms",   -10.0); // IMUは高レートなので幅を狭めにする

    prm_.delay_max_ms   = declare_parameter<double>("delay_max_ms",   +20.0);

    prm_.jitter_sigma_ms= declare_parameter<double>("jitter_sigma_ms", 2.0);

    prm_.burst_prob     = declare_parameter<double>("burst_prob", 0.01);

    prm_.burst_delay_ms = declare_parameter<double>("burst_delay_ms", 50.0);

    prm_.drop_prob      = declare_parameter<double>("drop_prob", 0.005);

    prm_.dup_prob       = declare_parameter<double>("dup_prob", 0.001);

    prm_.reorder_window_ms = declare_parameter<double>("reorder_window_ms", 10.0);

    prm_.skew_ppm       = declare_parameter<double>("skew_ppm", 100.0); // 安価なMEMSの温度ドリフト模擬

    prm_.hold_publish   = declare_parameter<bool>("hold_publish", true);

    prm_.stamp_only     = declare_parameter<bool>("stamp_only", false);

    prm_.seed           = static_cast<uint32_t>(declare_parameter<int>("seed", 0));

    in_topic_  = get_parameter("in_topic").as_string();

    out_topic_ = get_parameter("out_topic").as_string();

    rng_ = std::make_unique<psf::ChaosRNG>(prm_.seed);

    // Fix: reference point for elapsed-time-based skew, instead of the

    // per-message accumulator (see bug #1 above).

    start_time_ = this->get_clock()->now();

    sub_ = create_subscription<Imu>(in_topic_, rclcpp::SensorDataQoS(),

      std::bind(&ChaosInjectorImu::cb, this, std::placeholders::_1));

    pub_ = create_publisher<Imu>(out_topic_, 50);

    timer_ = create_wall_timer(std::chrono::milliseconds(5), // IMUは速いので5ms(200Hz)周期

              std::bind(&ChaosInjectorImu::drain, this));

    RCLCPP_WARN(get_logger(), "IMU chaos injector: %s -> %s", in_topic_.c_str(), out_topic_.c_str());

  }

private:

  psf::ChaosParams prm_;

  std::unique_ptr<psf::ChaosRNG> rng_;

  std::string in_topic_, out_topic_;

  rclcpp::Subscription<Imu>::SharedPtr sub_;

  rclcpp::Publisher<Imu>::SharedPtr pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::deque<ImuItem> buf_;

  rclcpp::Time start_time_;  // Fix: per-instance elapsed-time reference for skew (was a shared `static` accumulator)

  void cb(const Imu::SharedPtr msg) {

    if (rng_->bernoulli(prm_.drop_prob)) return;

    double d_ms = rng_->uniform(prm_.delay_min_ms, prm_.delay_max_ms)

                + rng_->normal(0.0, prm_.jitter_sigma_ms);

    if (rng_->bernoulli(prm_.burst_prob)) d_ms += prm_.burst_delay_ms;

    // Fix (bug #1a): skew now grows with REAL elapsed wall-clock time

    // since the node started, not with the sum of random per-message

    // delays -- so it no longer depends on the topic's publish rate.

    double elapsed_ms = (this->get_clock()->now() - start_time_).nanoseconds() / 1e6;

    double skew_ms = (prm_.skew_ppm * 1e-6) * elapsed_ms;

    // Fix (bug #1b): skew is applied to the REPORTED timestamp only.

    double stamped_total_ms = d_ms + skew_ms;

    auto out = std::make_shared<Imu>(*msg);

    rclcpp::Time orig_stamp = msg->header.stamp;

    rclcpp::Time new_stamp = rclcpp::Time((orig_stamp.nanoseconds() + static_cast<int64_t>(stamped_total_ms * 1e6)), RCL_ROS_TIME);

    out->header.stamp = new_stamp;

    if (prm_.stamp_only) {

      pub_->publish(*out);

      if (rng_->bernoulli(prm_.dup_prob)) pub_->publish(*out);

      return;

    }

    // Fix (bug #1b continued): the REAL scheduling delay uses d_ms alone

    // -- clock skew must not make the packet physically arrive later.

    double hold_ms = std::max(0.0, d_ms);

    rclcpp::Time due = this->get_clock()->now() + rclcpp::Duration::from_nanoseconds(static_cast<int64_t>(hold_ms * 1e6));

    buf_.push_back({out, due, orig_stamp});

  }

  void drain() {

    if (buf_.empty()) return;

    rclcpp::Time now = this->get_clock()->now();

    for (auto it = buf_.begin(); it != buf_.end();) {

      if (it->due <= now) {

        pub_->publish(*(it->msg));

        if (rng_->bernoulli(prm_.dup_prob)) pub_->publish(*(it->msg));

        it = buf_.erase(it);

      } else {

        ++it;

      }

    }

    if (buf_.size() >= 2) {

      double span_ms = (buf_.back().due - buf_.front().due).seconds() * 1000.0;

      if (span_ms > 0.0 && span_ms <= prm_.reorder_window_ms && rng_->bernoulli(0.3)) {

        size_t i = static_cast<size_t>(std::floor(rng_->uniform(0, (double)buf_.size())));

        size_t j = static_cast<size_t>(std::floor(rng_->uniform(0, (double)buf_.size())));

        // Fix (bug #2): swap only the DUE TIMES between the two slots so

        // the two messages actually trade delivery times -- swapping the

        // whole struct (as pasted) left publish order unchanged, since

        // drain() only ever looks at each item's own `due` value.

        std::swap(buf_[i].due, buf_[j].due);

      }

    }

  }

};

int main(int argc, char** argv){

  rclcpp::init(argc, argv);

  rclcpp::spin(std::make_shared<ChaosInjectorImu>());

  rclcpp::shutdown();

  return 0;

}
