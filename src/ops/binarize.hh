#include <stdint.h>


int8_t setBit(int8_t value, int n, int bitValue) {
    if (bitValue) {
        // Set nth bit to 1
        value |= (1 << n);
    } else {
        // Set nth bit to 0
        value &= ~(1 << n);
    }
    return value;
}

void binarize(int8_t* inputs, int8_t* outputs, int height, int width, int depth, int threshold, int out_vlen) {
    int8_t curr_value;
    int out_index;
    int out_offset
    int out_segment;
    int val_idx;
    int out_bit;

    for (int c = 0; c < depth; c++) {
        for (int h = 0; h < height; h ++) {
            for (int w = 0; w < width; w ++) {
                curr_value = inputs[c*height*width+h*width+w];
                out_index = (int) floor(c/out_vlen)
                out_offset = c % out_vlen;
                out_segment = (int) floor(out_offset / 8);
                out_bit = out_offset - out_segment * 8;

                val_idx = (out_index*height*width+h*width+w)*out_vlen/8+out_segment;
                
                outputs[val_idx] = setBit(outputs[val_idx],curr_value > threshold);
            }
        }
    }
}


