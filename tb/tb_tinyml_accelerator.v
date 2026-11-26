`timescale 1ns / 1ps

module tb_tinyml_accelerator;

    reg               clk;
    reg               rst;
    reg               start;
    reg               in_wr_en;
    reg  [2:0]        in_wr_addr;
    reg  signed [7:0] in_wr_data;
    wire              done;
    wire [1:0]        pred_class;
    wire signed [7:0] max_score;

    // Load all 20 test samples (20 samples * 8 features = 160 bytes)
    reg signed [7:0] test_mem [0:159];

    tinyml_accelerator_top dut (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .in_wr_en   (in_wr_en),
        .in_wr_addr (in_wr_addr),
        .in_wr_data (in_wr_data),
        .done       (done),
        .pred_class (pred_class),
        .max_score  (max_score)
    );

    // 100 MHz Clock (10 ns period)
    always #5 clk = ~clk;

    integer sample_idx, feat_idx, cycle_count;

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, tb_tinyml_accelerator);

        $readmemh("../quantized_data/test_inputs.mem", test_mem);

        clk        = 0;
        rst        = 1;
        start      = 0;
        in_wr_en   = 0;
        in_wr_addr = 0;
        in_wr_data = 0;

        #25 rst = 0;
        #10;

        for (sample_idx = 0; sample_idx < 20; sample_idx = sample_idx + 1) begin
            // 1. Load 8 input features into accelerator buffer
            for (feat_idx = 0; feat_idx < 8; feat_idx = feat_idx + 1) begin
                @(posedge clk);
                in_wr_en   <= 1'b1;
                in_wr_addr <= feat_idx[2:0];
                in_wr_data <= test_mem[(sample_idx * 8) + feat_idx];
            end
            @(posedge clk);
            in_wr_en <= 1'b0;

            // 2. Pulse start and count exact hardware clock cycles
            @(posedge clk);
            start       <= 1'b1;
            cycle_count <= 0;
            @(posedge clk);
            start       <= 1'b0;

            while (!done) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            $display("Sample %02d | Hardware Pred Class = %0d | Max Score = %4d | Latency = %0d cycles (%0d ns)",
                     sample_idx, pred_class, max_score, cycle_count, cycle_count * 10);

            #20;
        end

        $display("Simulation Complete! Compare Pred Class above with quantized_data/golden_outputs.txt");
        $finish;
    end

endmodule