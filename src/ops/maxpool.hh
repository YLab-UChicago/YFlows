
using namespace std;


void maxpool3d_2x2_s2_8b(int8_t* inputs, int8_t* outputs,
                     int height, int width, int depth){

    int8x16_t vec_00;
    int8x16_t vec_01;
    int8x16_t vec_10;
    int8x16_t vec_11;
    
    for (int c = 0; c < depth; c ++) {
        for (int h = 0; h < height; h ++) {
            for (int w = 0; w < width; w ++) {

                vec_00 = vld1q_s8(inputs + (c*height*width+h*width+w)*16);
                vec_01 = vld1q_s8(inputs + ((c*height*width+h*width+w)*2+1)*16);
                vec_10 = vld1q_s8(inputs + (c*height*width+(h+1)*width+w)*16);
                vec_11 = vld1q_s8(inputs + ((c*height*width+(h+1)*width+w)*2+1)*16);

                vst1q_s8(outputs,vmaxq_s8(vmaxq_s8(vec_00,vec_01),vmaxq_s8(vec_10,vec_11)));
            }
        }
    }
    return;
}


void maxpool3d_2x2_s2_1b(int16_t* inputs, int16_t* outputs,
                     int height, int width, int depth){

    int16x8_t vec_00;
    int16x8_t vec_01;
    int16x8_t vec_10;
    int16x8_t vec_11;
    
    for (int c = 0; c < depth / 16; c ++) {
        for (int h = 0; h < height; h ++) {
            for (int w = 0; w < width; w ++) {

                __asm__ volatile ("pld [%0]" : : "r"(inputs + (c*height*width+h*width+w+5)*8) : "memory");

                vec_00 = vld1q_s16(inputs + (c*height*width+h*width+w)*8);
                vec_01 = vld1q_s16(inputs + ((c*height*width+h*width+w)*2+1)*8);
                vec_10 = vld1q_s16(inputs + (c*height*width+(h+1)*width+w)*8);
                vec_11 = vld1q_s16(inputs + ((c*height*width+(h+1)*width+w)*2+1)*8);

                vst1q_s16(outputs,vorrq_s16(vmaxq_s16(vec_00,vec_01),vorrq_s16(vec_10,vec_11)));
            }
        }
    }
    return;

}


void avgpool3d_2x2_s2_8b(int16_t* inputs, int16_t* outputs,
                     int height, int width, int depth){

    int16x8_t vec_00;
    int16x8_t vec_01;
    int16x8_t vec_10;
    int16x8_t vec_11;
    
    for (int c = 0; c < depth; c ++) {
        for (int h = 0; h < height; h ++) {
            for (int w = 0; w < width; w ++) {
                
                __asm__ volatile ("pld [%0]" : : "r"(inputs + (c*height*width+h*width+w+5)*8) : "memory");

                vec_00 = vld1q_s16(inputs + (c*height*width+h*width+w)*8);
                vec_01 = vld1q_s16(inputs + ((c*height*width+h*width+w)*2+1)*8);
                vec_10 = vld1q_s16(inputs + (c*height*width+(h+1)*width+w)*8);
                vec_11 = vld1q_s16(inputs + ((c*height*width+(h+1)*width+w)*2+1)*8);

                vst1q_s16(outputs,vshrq_n_s16((vaddq_s16(vaddq_s16(vec_00,vec_01),vaddq_s16(vec_10,vec_11))),16));
            }
        }


    }

    return;

}