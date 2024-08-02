#ifndef CONV_H
#define CONV_H

typedef struct {
        // Parameters
        short input_channels;
        short output_channels;
        short kernel_xy;
        short stride_xy;
        short padding_xy;
        float* weights;
} Conv_Struct;

Conv_Struct conv_constructor(short input_channels, short output_channels,
                             short kernel_xy, short stride_xy, short padding_xy);

#endif
