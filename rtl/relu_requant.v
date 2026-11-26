`timescale 1ns / 1ps

module relu_requant #(
    parameter SHIFT = 7
)(
    input  wire signed [31:0] acc_in,
    input  wire signed [31:0] bias_in,
    input  wire               apply_relu, // 1 for Hidden Layer, 0 for Output Layer
    output reg  signed [7:0]  data_out
);

    wire signed [31:0] sum_with_bias;
    wire signed [31:0] shifted;

    assign sum_with_bias = acc_in + bias_in;
    assign shifted       = sum_with_bias >>> SHIFT;

    always @(*) begin
        if (apply_relu && (sum_with_bias < 32'sd0)) begin
            data_out = 8'sd0;
        end else if (shifted > 32'sd127) begin
            data_out = 8'sd127;
        end else if (shifted < -32'sd128) begin
            data_out = -8'sd128;
        end else begin
            data_out = shifted[7:0];
        end
    end

endmodule