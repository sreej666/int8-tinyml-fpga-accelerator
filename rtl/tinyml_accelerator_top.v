`timescale 1ns / 1ps

module tinyml_accelerator_top #(
    parameter INPUT_SIZE  = 8,
    parameter HIDDEN_SIZE = 16,
    parameter OUTPUT_SIZE = 4,
    parameter SHIFT_L1    = 7,
    parameter SHIFT_L2    = 7
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,
    input  wire              in_wr_en,
    input  wire [2:0]        in_wr_addr,
    input  wire signed [7:0] in_wr_data,
    output reg               done,
    output reg  [1:0]        pred_class,
    output reg  signed [7:0] max_score
);

    // Parameter Memories
    reg signed [7:0]  w1_rom [0:(INPUT_SIZE*HIDDEN_SIZE)-1];
    reg signed [31:0] b1_rom [0:HIDDEN_SIZE-1];
    reg signed [7:0]  w2_rom [0:(HIDDEN_SIZE*OUTPUT_SIZE)-1];
    reg signed [31:0] b2_rom [0:OUTPUT_SIZE-1];

    initial begin
        $readmemh("../quantized_data/weights_l1.mem", w1_rom);
        $readmemh("../quantized_data/biases_l1.mem",  b1_rom);
        $readmemh("../quantized_data/weights_l2.mem", w2_rom);
        $readmemh("../quantized_data/biases_l2.mem",  b2_rom);
    end

    // Feature & Hidden Activation Buffers
    reg signed [7:0] input_buf  [0:INPUT_SIZE-1];
    reg signed [7:0] hidden_buf [0:HIDDEN_SIZE-1];

    always @(posedge clk) begin
        if (in_wr_en) begin
            input_buf[in_wr_addr] <= in_wr_data;
        end
    end

    // FSM States
    localparam S_IDLE     = 3'd0,
               S_L1_MAC   = 3'd1,
               S_L1_DRAIN = 3'd2,
               S_L1_ACT   = 3'd3,
               S_L2_MAC   = 3'd4,
               S_L2_DRAIN = 3'd5,
               S_L2_ACT   = 3'd6,
               S_DONE     = 3'd7;

    reg [2:0] state;
    reg [4:0] neuron_idx;
    reg [4:0] input_idx;

    // Combinational MAC Control (Zero-skew alignment with input_idx)
    wire mac_clear = (state == S_IDLE) || (state == S_L1_ACT) || (state == S_L2_ACT);
    wire mac_en    = (state == S_L1_MAC) || (state == S_L2_MAC);

    reg  signed [7:0]  mac_a;
    reg  signed [7:0]  mac_b;
    wire signed [31:0] mac_acc;

    mac_int8 u_mac (
        .clk       (clk),
        .rst       (rst),
        .clear_acc (mac_clear),
        .en        (mac_en),
        .a         (mac_a),
        .b         (mac_b),
        .acc       (mac_acc)
    );

    // Layer-Specific Re-quantization Datapaths
    reg  signed [31:0] current_bias;
    wire signed [7:0]  requant_l1_out;
    wire signed [7:0]  requant_l2_out;

    relu_requant #(.SHIFT(SHIFT_L1)) u_requant_l1 (
        .acc_in     (mac_acc),
        .bias_in    (current_bias),
        .apply_relu (1'b1),
        .data_out   (requant_l1_out)
    );

    relu_requant #(.SHIFT(SHIFT_L2)) u_requant_l2 (
        .acc_in     (mac_acc),
        .bias_in    (current_bias),
        .apply_relu (1'b0),
        .data_out   (requant_l2_out)
    );

    // Combinational Data Routing
    always @(*) begin
        mac_a        = 8'sd0;
        mac_b        = 8'sd0;
        current_bias = 32'sd0;

        if (state == S_L1_MAC) begin
            mac_a = input_buf[input_idx[2:0]];
            mac_b = w1_rom[(neuron_idx * INPUT_SIZE) + input_idx];
        end else if (state == S_L1_ACT) begin
            current_bias = b1_rom[neuron_idx[3:0]];
        end else if (state == S_L2_MAC) begin
            mac_a = hidden_buf[input_idx[3:0]];
            mac_b = w2_rom[(neuron_idx * HIDDEN_SIZE) + input_idx];
        end else if (state == S_L2_ACT) begin
            current_bias = b2_rom[neuron_idx[1:0]];
        end
    end

    // Master Control FSM
    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            neuron_idx <= 5'd0;
            input_idx  <= 5'd0;
            done       <= 1'b0;
            pred_class <= 2'd0;
            max_score  <= -8'sd128;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        neuron_idx <= 5'd0;
                        input_idx  <= 5'd0;
                        state      <= S_L1_MAC;
                    end
                end

                S_L1_MAC: begin
                    if (input_idx == INPUT_SIZE - 1) begin
                        state <= S_L1_DRAIN;
                    end else begin
                        input_idx <= input_idx + 5'd1;
                    end
                end

                S_L1_DRAIN: begin
                    state <= S_L1_ACT;
                end

                S_L1_ACT: begin
                    hidden_buf[neuron_idx[3:0]] <= requant_l1_out;
                    input_idx <= 5'd0;
                    if (neuron_idx == HIDDEN_SIZE - 1) begin
                        neuron_idx <= 5'd0;
                        state      <= S_L2_MAC;
                    end else begin
                        neuron_idx <= neuron_idx + 5'd1;
                        state      <= S_L1_MAC;
                    end
                end

                S_L2_MAC: begin
                    if (input_idx == HIDDEN_SIZE - 1) begin
                        state <= S_L2_DRAIN;
                    end else begin
                        input_idx <= input_idx + 5'd1;
                    end
                end

                S_L2_DRAIN: begin
                    state <= S_L2_ACT;
                end

                S_L2_ACT: begin
                    if (neuron_idx == 5'd0 || (requant_l2_out > max_score)) begin
                        max_score  <= requant_l2_out;
                        pred_class <= neuron_idx[1:0];
                    end

                    input_idx <= 5'd0;
                    if (neuron_idx == OUTPUT_SIZE - 1) begin
                        state <= S_DONE;
                    end else begin
                        neuron_idx <= neuron_idx + 5'd1;
                        state      <= S_L2_MAC;
                    end
                end

                S_DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule