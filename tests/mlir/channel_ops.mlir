// RUN: cppfort-opt %s | FileCheck %s
// Test MLIR operations for channelized concurrency
// Tests channel, send, recv operations with escape analysis validation
//
// NOTE: This test requires MLIR infrastructure to be built. The test file
// documents expected behavior and structure for channel ops.
// Execution requires: cppfort-opt tool with Cpp2FIRDialect support
//
// Current status: Test structure defined, execution pending MLIR build

module {
  // Test 1: Basic channel creation
  // CHECK-LABEL: @create_channel
  cpp2fir.func @create_channel() -> !cpp2.channel<i32> {
    // CHECK: cpp2fir.channel_create i32
    %0 = cpp2fir.channel_create i32 -> !cpp2.channel<i32>
    // CHECK: return
    cpp2fir.return %0 : !cpp2.channel<i32>
  }

  // Test 2: Unbuffered channel (synchronous)
  // CHECK-LABEL: @unbuffered_channel
  cpp2fir.func @unbuffered_channel() -> !cpp2.channel<f64> {
    // CHECK: cpp2fir.channel_create f64 buffer 0
    %0 = cpp2fir.channel_create f64 buffer 0 -> !cpp2.channel<f64>
    cpp2fir.return %0 : !cpp2.channel<f64>
  }

  // Test 3: Buffered channel with capacity
  // CHECK-LABEL: @buffered_channel
  cpp2fir.func @buffered_channel() -> !cpp2.channel<!cpp2.string> {
    // CHECK: cpp2fir.channel_create !cpp2.string buffer 10
    %0 = cpp2fir.channel_create !cpp2.string buffer 10 -> !cpp2.channel<!cpp2.string>
    cpp2fir.return %0 : !cpp2.channel<!cpp2.string>
  }

  // Test 4: Channel send operation
  // CHECK-LABEL: @channel_send_op
  cpp2fir.func @channel_send_op(%chan: !cpp2.channel<i32>, %val: i32) {
    // CHECK: cpp2fir.channel_send %{{.*}} to %{{.*}} : i32 to !cpp2.channel<i32>
    cpp2fir.channel_send %val to %chan : i32 to !cpp2.channel<i32>
    // CHECK: return
    cpp2fir.return
  }

  // Test 5: Async send with synchronization
  // CHECK-LABEL: @async_channel_send
  cpp2fir.func @async_channel_send(%chan: !cpp2.channel<f64>, %val: f64) {
    // CHECK: cpp2fir.channel_send %{{.*}} to %{{.*}} async sync "send_complete"
    cpp2fir.channel_send %val to %chan async sync "send_complete" : f64 to !cpp2.channel<f64>
    // CHECK: cpp2fir.sync_point "send_complete"
    cpp2fir.sync_point "send_complete"
    // CHECK: return
    cpp2fir.return
  }

  // Test 6: Channel receive operation
  // CHECK-LABEL: @channel_recv_op
  cpp2fir.func @channel_recv_op(%chan: !cpp2.channel<!cpp2.string>) -> !cpp2.string {
    // CHECK: %{{.*}} = cpp2fir.channel_recv from %{{.*}} : !cpp2.channel<!cpp2.string> -> !cpp2.string
    %0 = cpp2fir.channel_recv from %chan : !cpp2.channel<!cpp2.string> -> !cpp2.string
    // CHECK: return %{{.*}} : !cpp2.string
    cpp2fir.return %0 : !cpp2.string
  }

  // Test 7: Select operation (non-deterministic choice)
  // CHECK-LABEL: @channel_select
  cpp2fir.func @channel_select(%chan1: !cpp2.channel<i32>, %chan2: !cpp2.channel<i32>) -> i32 {
    // CHECK: %{{.*}} = cpp2fir.channel_select
    // CHECK-SAME: case %{{.*}} from %{{.*}} : {{.*}} -> ^bb1
    // CHECK-SAME: case %{{.*}} from %{{.*}} : {{.*}} -> ^bb2
    %0 = cpp2fir.channel_select
      case %val1 from %chan1 : i32 -> ^bb1
      case %val2 from %chan2 : i32 -> ^bb2

    // CHECK: ^bb1:
    // CHECK: cpp2fir.return %val1
    ^bb1(%val1: i32):
      cpp2fir.return %val1 : i32

    // CHECK: ^bb2:
    // CHECK: cpp2fir.return %val2
    ^bb2(%val2: i32):
      cpp2fir.return %val2 : i32
  }

  // Test 8: Channel close operation
  // CHECK-LABEL: @channel_close
  cpp2fir.func @channel_close(%chan: !cpp2.channel<f64>) {
    // CHECK: cpp2fir.channel_close %{{.*}} : !cpp2.channel<f64>
    cpp2fir.channel_close %chan : !cpp2.channel<f64>
    // CHECK: return
    cpp2fir.return
  }

  // Test 9: Channel as first-class value (pass to function)
  // CHECK-LABEL: @channel_as_parameter
  cpp2fir.func @channel_as_parameter(%chan: !cpp2.channel<i32>, %data: i32) {
    // CHECK: cpp2fir.call @process_channel(%{{.*}}) : (!cpp2.channel<i32>) -> ()
    cpp2fir.call @process_channel(%chan) : (!cpp2.channel<i32>) -> ()
    cpp2fir.return
  }

  // Helper function that takes a channel
  cpp2fir.func private @process_channel(!cpp2.channel<i32>) -> ()

  // Test 10: Channel in struct
  // CHECK-LABEL: @channel_in_struct
  cpp2fir.func @channel_in_struct() {
    // CHECK: %{{.*}} = cpp2fir.alloca !cpp2.struct<"Server", {{.*}}>
    %0 = cpp2fir.alloca !cpp2.struct<"Server", #cpp2.rec<
      request_chan: !cpp2.channel<Request>,
      response_chan: !cpp2.channel<Response>
    >>
    cpp2fir.return
  }

  // Test 11: Range-based channel iteration
  // CHECK-LABEL: @channel_range
  cpp2fir.func @channel_range(%chan: !cpp2.channel<i32>) {
    // CHECK: cpp2fir.channel_range %{{.*}} {
    cpp2fir.channel_range %chan {
      // CHECK: ^bb1(%{{.*}}: i32):
    ^bb1(%value: i32):
      // Process value
      // CHECK: cpp2fir.yield
      cpp2fir.yield
    }
    // CHECK: return
    cpp2fir.return
  }

  // Test 12: Channel with escape analysis annotations
  // CHECK-LABEL: @channel_with_escape_analysis
  cpp2fir.func @channel_with_escape_analysis(%chan: !cpp2.channel<!cpp2.handle>) {
    // CHECK: %{{.*}} = cpp2fir.channel_recv from %{{.*}} escape !cpp2.EscapeToChannel
    %0 = cpp2fir.channel_recv from %chan
      {escape = !cpp2.EscapeToChannel} : !cpp2.channel<!cpp2.handle> -> !cpp2.handle

    // CHECK: cpp2fir.channel_send %{{.*}} to %{{.*}} escape !cpp2.NoEscape
    cpp2fir.channel_send %0 to %chan
      {escape = !cpp2.NoEscape} : !cpp2.handle to !cpp2.channel<!cpp2.handle>

    cpp2fir.return
  }

  // Test 13: Multiple channel operations with sync points
  // CHECK-LABEL: @multi_channel_sync
  cpp2fir.func @multi_channel_sync(%ch1: !cpp2.channel<i32>, %ch2: !cpp2.channel<f64>) {
    // Send to both channels asynchronously
    // CHECK: cpp2fir.channel_send %{{.*}} to %{{.*}} async
    %c1 = cpp2fir.constant 42 : i32
    cpp2fir.channel_send %c1 to %ch1 async sync "sync1" : i32 to !cpp2.channel<i32>

    // CHECK: cpp2fir.channel_send %{{.*}} to %{{.*}} async
    %c2 = cpp2fir.constant 3.14 : f64
    cpp2fir.channel_send %c2 to %ch2 async sync "sync2" : f64 to !cpp2.channel<f64>

    // Wait for both sends to complete
    // CHECK: cpp2fir.sync_point "sync1"
    // CHECK: cpp2fir.sync_point "sync2"
    cpp2fir.sync_point "sync1"
    cpp2fir.sync_point "sync2"

    cpp2fir.return
  }

  // Test 14: Channel capacity query
  // CHECK-LABEL: @channel_capacity
  cpp2fir.func @channel_capacity(%chan: !cpp2.channel<i32>) -> i32 {
    // CHECK: %{{.*}} = cpp2fir.channel_capacity %{{.*}} : !cpp2.channel<i32>
    %0 = cpp2fir.channel_capacity %chan : !cpp2.channel<i32>
    // CHECK: return %{{.*}} : i32
    cpp2fir.return %0 : i32
  }

  // Test 15: Channel length query (number of pending items)
  // CHECK-LABEL: @channel_length
  cpp2fir.func @channel_length(%chan: !cpp2.channel<i32>) -> i32 {
    // CHECK: %{{.*}} = cpp2fir.channel_length %{{.*}} : !cpp2.channel<i32>
    %0 = cpp2fir.channel_length %chan : !cpp2.channel<i32>
    // CHECK: return %{{.*}} : i32
    cpp2fir.return %0 : i32
  }
}
