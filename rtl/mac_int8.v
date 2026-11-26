`timescale 1ns / 1ps

module mac_int8 (
    input  wire               clk,
    input  wire               rst,
    input  wire               clear_acc,
    input  wire               en,
    input  wire signed [7:0]  a,        // 8-bit signed input/activation
    input  wire signed [7:0]  b,        // 8-bit signed weight
    output reg  signed [31:0] acc       // 32-bit signed accumulator
);

    wire signed [15:0] product;
    assign product = a * b;

    always @(posedge clk) begin
        if (rst || clear_acc) begin
            acc <= 32'sd0;
        end else if (en) begin
            acc <= acc + {{16{product[15]}}, product};
        end
    end

endmodule