#ifndef _CONV2d_3x3_
#define _CONV2d_3x3_


#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <arm_neon.h>
#include <algorithm>
#include <omp.h>
using namespace std;



int conv2d_3x3_vl128_s8(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {

    int filter_height = 3;
    int filter_width = 3;
    int out_vlen = 128;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;
    int out_idx;

    int8x16_t output;

    int8x16_t input_cache_0;
    int8x16_t input_cache_1;
    int8x16_t input_cache_2;
    int8x16_t input_cache_3;
    int8x16_t input_cache_4;
    int8x16_t input_cache_5;
    
    int8x16_t weight_cache_0;
    int8x16_t weight_cache_1;
    int8x16_t weight_cache_2;
    int8x16_t weight_cache_3;
    int8x16_t weight_cache_4;
    int8x16_t weight_cache_5;
    int8x16_t weight_cache_6;
    int8x16_t weight_cache_7;
    int8x16_t weight_cache_8;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int k = 0; k < num_filters; k += k_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 16) {
            input_cache_0 = vld1q_s8((const int8_t*)inputs+_c*height*width+0);
            input_cache_1 = vld1q_s8((const int8_t*)inputs+_c*height*width+16);
            input_cache_2 = vld1q_s8((const int8_t*)inputs+_c*height*width+width*16);
            input_cache_3 = vld1q_s8((const int8_t*)inputs+_c*height*width+width*16+16);
            input_cache_4 = vld1q_s8((const int8_t*)inputs+_c*height*width+width*32+0);
            input_cache_5 = vld1q_s8((const int8_t*)inputs+_c*height*width+width*32+16);

                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {
                    weight_cache_0 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+0);
                    weight_cache_1 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+16);
                    weight_cache_2 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32);
                    weight_cache_3 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+48);
                    weight_cache_4 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64);
                    weight_cache_5 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+80);
                    weight_cache_6 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+96);
                    weight_cache_7 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+112);
                    weight_cache_8 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+128);
                    

                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    // __builtin_prefetch(inputs + _c * height * width + ((_h ) * width + _w) * 16, 0, 3);
                                    output = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_0,weight_cache_0,output);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_1,weight_cache_1,output);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_0 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_0,weight_cache_2,output);
    

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_2,weight_cache_3,output);
                                        
                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_3,weight_cache_4,output);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_2 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_2,weight_cache_5,output);
                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_4,weight_cache_6,output);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_5,weight_cache_7,output);
                                
                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input_cache_4 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_4,weight_cache_8,output);
                                    

                                    out_idx = ((int)(_k/floor(out_vlen/8))*out_height*out_width+_h*out_width+_w)*(out_vlen/8)+(int)_k%(out_vlen/8);
                                    outputs[out_idx] += vaddvq_s8(output);
                                    if (_c+16 >= num_channel-1) {
                                        outputs[out_idx] = max(outputs[out_idx], (int8_t)0);
                                    }

                                    _w ++;
                                    output = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_1,weight_cache_0,output);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_0,weight_cache_1,output);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_1 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_1,weight_cache_2,output);
                                        

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_3,weight_cache_3,output);

                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_2,weight_cache_4,output);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_3 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_3,weight_cache_5,output);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output = vmlaq_s8(input_cache_5,weight_cache_6,output);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    output = vmlaq_s8(input_cache_4,weight_cache_7,output);

                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input_cache_5 = vld1q_s8((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*16);
                                    output = vmlaq_s8(input_cache_5,weight_cache_8,output);

                                    // printf("3\n");
                                    out_idx = ((int)(_k/floor(out_vlen/8))*out_height*out_width+_h*out_width+_w)*(out_vlen/8)+(int)_k%(out_vlen/8);
                                    outputs[out_idx] += vaddvq_s8(output);
                                    if (_c+16 >= num_channel-1) {
                                        outputs[out_idx] = max(outputs[out_idx], (int8_t)0);
                                    }
                                    // printf("4\n");
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}

int conv2d_3x3_vl256_s8(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {
    int filter_height = 3;
    int filter_width = 3;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;
    int out_vlen = 256;
    int out_idx;

    int8x16x2_t input;
    int8x16x2_t output;

    int8x16x2_t input_cache_0;
    int8x16x2_t input_cache_1;
    int8x16x2_t input_cache_2;
    int8x16x2_t input_cache_3;
    int8x16x2_t input_cache_4;
    int8x16x2_t input_cache_5;
    
    int8x16x2_t weight_cache_0;
    int8x16x2_t weight_cache_1;
    int8x16x2_t weight_cache_2;
    int8x16x2_t weight_cache_3;
    int8x16x2_t weight_cache_4;
    int8x16x2_t weight_cache_5;
    int8x16x2_t weight_cache_6;
    int8x16x2_t weight_cache_7;
    int8x16x2_t weight_cache_8;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 32) {
            input_cache_0 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+width*32*0+32*0);
            input_cache_1 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+width*32*0+32*1);
            input_cache_2 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+width*32*1+32*0);
            input_cache_3 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+width*32*1+32*1);
            input_cache_4 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+width*32*2+32*0);

            for (int k = 0; k < num_filters; k += k_block) {
                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {
                    weight_cache_0 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*0);
                    weight_cache_1 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*1);
                    weight_cache_2 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*2);
                    weight_cache_3 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*3);
                    weight_cache_4 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*4);
                    weight_cache_5 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*5);
                    weight_cache_6 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*6);
                    weight_cache_7 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*7);
                    weight_cache_8 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+32*8);
                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    // __builtin_prefetch(inputs + _c * height * width + ((_h ) * width + _w) * 16, 0, 3);
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_0.val[0],weight_cache_0.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_0.val[1],weight_cache_0.val[1],output.val[1]);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output.val[0] = vmlaq_s8(input_cache_1.val[0],weight_cache_1.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_1.val[1],weight_cache_1.val[1],output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_0 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input_cache_0.val[0],weight_cache_2.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_0.val[1],weight_cache_2.val[1],output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_2.val[0],weight_cache_3.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_2.val[1],weight_cache_3.val[1],output.val[1]);
                                        
                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output.val[0] = vmlaq_s8(input_cache_3.val[0],weight_cache_4.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_3.val[1],weight_cache_4.val[1],output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_2 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input_cache_2.val[0],weight_cache_5.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_2.val[1],weight_cache_5.val[1],output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_4.val[0],weight_cache_6.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_4.val[1],weight_cache_6.val[1],output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_7.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_7.val[1],output.val[1]);
                                
                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_8.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_8.val[1],output.val[1]);
                                    out_idx = ((int)(_k/floor(out_vlen/8))*out_height*out_width+_h*out_width+_w)*(out_vlen/8)+(int)_k%(out_vlen/8);
                                    
                                    outputs[out_idx] += vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1]);
                                    if (_c+32 >= num_channel-1) {
                                        outputs[out_idx] = max(outputs[out_idx], (int8_t)0);
                                    }

                                    _w ++;
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_1.val[0],weight_cache_0.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_1.val[1],weight_cache_0.val[1],output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output.val[0] = vmlaq_s8(input_cache_0.val[0],weight_cache_1.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_0.val[1],weight_cache_1.val[1],output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_1 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input_cache_1.val[0],weight_cache_2.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_1.val[1],weight_cache_2.val[1],output.val[1]);
                                        

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_3.val[0],weight_cache_3.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_3.val[1],weight_cache_3.val[1],output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output.val[0]= vmlaq_s8(input_cache_2.val[0],weight_cache_4.val[0],output.val[0]);
                                    output.val[1]= vmlaq_s8(input_cache_2.val[1],weight_cache_4.val[1],output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_3 = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input_cache_3.val[0],weight_cache_5.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_3.val[1],weight_cache_5.val[1],output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output.val[0] = vmlaq_s8(input_cache_5.val[0],weight_cache_6.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input_cache_5.val[1],weight_cache_6.val[1],output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_7.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_7.val[1],output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*32);
                                    output.val[0]= vmlaq_s8(input.val[0],weight_cache_8.val[0],output.val[0]);
                                    output.val[1]= vmlaq_s8(input.val[1],weight_cache_8.val[1],output.val[1]);
                                    
                                    out_idx = ((int)(_k/floor(out_vlen/8))*out_height*out_width+_h*out_width+_w)*(out_vlen/8)+(int)_k%(out_vlen/8);
                                    outputs[out_idx] += vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1]);
                                    if (_c+32 >= num_channel-1) {
                                        outputs[out_idx] = max(outputs[out_idx], (int8_t)0);
                                    }
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}

int conv2d_3x3_vl512_s8(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {
    int filter_height = 3;
    int filter_width = 3;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;
    int out_vlen = 256;
    int out_idx;

    int8x16x4_t input;
    int8x16x4_t output;
    int8x16x4_t weight;
    
    int8x16x4_t weight_cache_0;
    int8x16x4_t weight_cache_1;
    int8x16x4_t weight_cache_2;
    int8x16x4_t weight_cache_3;
    int8x16x4_t weight_cache_4;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 64) {
            for (int k = 0; k < num_filters; k += k_block) {
                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {

                    weight_cache_0 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64*0);
                    weight_cache_1 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64*1);
                    weight_cache_2 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64*2);
                    weight_cache_3 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64*3);
                    weight_cache_4 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c*filter_height*filter_width+64*4);

                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    // __builtin_prefetch(inputs + _c * height * width + ((_h ) * width + _w) * 16, 0, 3);
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);
                                    output.val[2] = vdupq_n_s8(0);
                                    output.val[3] = vdupq_n_s8(0);
                                    
                                    i = 0;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_0.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_0.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_0.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_0.val[3],output.val[3]);
                                    
                                    i = 0;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_1.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_1.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_1.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_1.val[3],output.val[3]);

                                    i = 0;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_2.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_2.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_2.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_2.val[3],output.val[3]);

                                    i = 1;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_3.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_3.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_3.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_3.val[3],output.val[3]);                                   
                                        
                                    i = 1;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_4.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_4.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_4.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_4.val[3],output.val[3]);

                                    i = 1;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);

                                    i = 2;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);

                                    i = 2;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);
                                
                                    i = 2;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);
                                    outputs[_k*out_height*out_width+_h*out_width+_w] += (int8_t) max(0,(int)(vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1])+vaddvq_s8(output.val[2])+vaddvq_s8(output.val[3])));
                                    out_idx = ((int)(_k/floor(out_vlen/8))*out_height*out_width+_h*out_width+_w)*(out_vlen/8)+(int)_k%(out_vlen/8);
                                    outputs[out_idx] += vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1])+vaddvq_s8(output.val[2])+vaddvq_s8(output.val[3]);
                                    if (_c+64 >= num_channel-1) {
                                        outputs[out_idx] = max(outputs[out_idx], (int8_t)0);
                                    }
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}



int conv2d_3x3_vl128_s1(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {
    int filter_height = 3;
    int filter_width = 3;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;

    int8x16_t output;

    int8x16_t input_cache_0;
    int8x16_t input_cache_1;
    int8x16_t input_cache_2;
    int8x16_t input_cache_3;
    int8x16_t input_cache_4;
    int8x16_t input_cache_5;
    
    int8x16_t weight_cache_0;
    int8x16_t weight_cache_1;
    int8x16_t weight_cache_2;
    int8x16_t weight_cache_3;
    int8x16_t weight_cache_4;
    int8x16_t weight_cache_5;
    int8x16_t weight_cache_6;
    int8x16_t weight_cache_7;
    int8x16_t weight_cache_8;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 128) {
            input_cache_0 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+0);
            input_cache_1 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+16);
            input_cache_2 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+width*16);
            input_cache_3 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+width*16+16);
            input_cache_4 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+width*32+0);
            input_cache_5 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+width*32+16);

            for (int k = 0; k < num_filters; k += k_block) {
                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {
                    weight_cache_0 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+0);
                    weight_cache_1 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+16);
                    weight_cache_2 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32);
                    weight_cache_3 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+48);
                    weight_cache_4 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64);
                    weight_cache_5 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+80);
                    weight_cache_6 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+96);
                    weight_cache_7 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+112);
                    weight_cache_8 = vld1q_s8((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+128);
                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    output = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0,weight_cache_0)),output);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1,weight_cache_1)),output);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_0 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0,weight_cache_2)),output);
    
                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2,weight_cache_3)),output);
                                        
                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3,weight_cache_4)),output);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_2 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2,weight_cache_5)),output);
                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_4,weight_cache_6)),output);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_5,weight_cache_7)),output);
                                
                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input_cache_4 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_4,weight_cache_8)),output);
                                    outputs[_k*out_height*out_width+_h*out_width+_w] += vaddvq_s8(output);
                                    
                                    _w ++;
                                    output = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1,weight_cache_0)),output);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0,weight_cache_1)),output);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_1 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1,weight_cache_2)),output);
                                        
                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3,weight_cache_3)),output);

                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2,weight_cache_4)),output);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_3 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3,weight_cache_5)),output);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_5,weight_cache_6)),output);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_4,weight_cache_7)),output);

                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input_cache_5 = vld1q_s8((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*16);
                                    output = vaddq_s8(vcntq_s8(veorq_s8(input_cache_5,weight_cache_8)),output);

                                    outputs[_k*out_height*out_width+_h*out_width+_w] += (int8_t) max(0,(int)vaddvq_s8(output));
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}

int conv2d_3x3_vl256_s1(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {
    int filter_height = 3;
    int filter_width = 3;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;

    int8x16x2_t input;
    int8x16x2_t output;

    int8x16x2_t input_cache_0;
    int8x16x2_t input_cache_1;
    int8x16x2_t input_cache_2;
    int8x16x2_t input_cache_3;
    int8x16x2_t input_cache_4;
    int8x16x2_t input_cache_5;
    
    int8x16x2_t weight_cache_0;
    int8x16x2_t weight_cache_1;
    int8x16x2_t weight_cache_2;
    int8x16x2_t weight_cache_3;
    int8x16x2_t weight_cache_4;
    int8x16x2_t weight_cache_5;
    int8x16x2_t weight_cache_6;
    int8x16x2_t weight_cache_7;
    int8x16x2_t weight_cache_8;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 256) {
            input_cache_0 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+width*32*0+32*0);
            input_cache_1 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+width*32*0+32*1);
            input_cache_2 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+width*32*1+32*0);
            input_cache_3 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+width*32*1+32*1);
            input_cache_4 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+width*32*2+32*0);

            for (int k = 0; k < num_filters; k += k_block) {
                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {
                    weight_cache_0 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*0);
                    weight_cache_1 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*1);
                    weight_cache_2 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*2);
                    weight_cache_3 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*3);
                    weight_cache_4 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*4);
                    weight_cache_5 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*5);
                    weight_cache_6 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*6);
                    weight_cache_7 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*7);
                    weight_cache_8 = vld1q_s8_x2((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+32*8);
                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    // __builtin_prefetch(inputs + _c * height * width + ((_h ) * width + _w) * 16, 0, 3);
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[0],weight_cache_0.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[1],weight_cache_0.val[1])),output.val[1]);
                                    
                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[0],weight_cache_1.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[1],weight_cache_1.val[1])),output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input_cache_0 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[0],weight_cache_2.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[1],weight_cache_2.val[1])),output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[0],weight_cache_3.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[1],weight_cache_3.val[1])),output.val[1]);
                                        
                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[0],weight_cache_4.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[1],weight_cache_4.val[1])),output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input_cache_2 = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[0],weight_cache_5.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[1],weight_cache_5.val[1])),output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_4.val[0],weight_cache_6.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_4.val[1],weight_cache_6.val[1])),output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input.val[0],weight_cache_7.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input.val[1],weight_cache_7.val[1])),output.val[1]);
                                
                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input.val[0],weight_cache_8.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input.val[1],weight_cache_8.val[1])),output.val[1]);
                                    outputs[_k*out_height*out_width+_h*out_width+_w] += (int8_t) max(0,(int)(vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1])));
                                    
                                    _w ++;
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);

                                    input_h = _h+0;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[0],weight_cache_0.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[1],weight_cache_0.val[1])),output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+1;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[0],weight_cache_1.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_0.val[1],weight_cache_1.val[1])),output.val[1]);

                                    input_h = _h+0;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[0],weight_cache_2.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_1.val[1],weight_cache_2.val[1])),output.val[1]);
                                        

                                    input_h = _h+1;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[0],weight_cache_3.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[1],weight_cache_3.val[1])),output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+1;
                                    output.val[0]= vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[0],weight_cache_4.val[0])),output.val[0]);
                                    output.val[1]= vaddq_s8(vcntq_s8(veorq_s8(input_cache_2.val[1],weight_cache_4.val[1])),output.val[1]);

                                    input_h = _h+1;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[0],weight_cache_5.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_3.val[1],weight_cache_5.val[1])),output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+0;
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_5.val[0],weight_cache_6.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input_cache_5.val[1],weight_cache_6.val[1])),output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+1;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0] = vaddq_s8(vcntq_s8(veorq_s8(input.val[0],weight_cache_7.val[0])),output.val[0]);
                                    output.val[1] = vaddq_s8(vcntq_s8(veorq_s8(input.val[1],weight_cache_7.val[1])),output.val[1]);

                                    input_h = _h+2;
                                    input_w = _w+2;
                                    input = vld1q_s8_x2((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*32);
                                    output.val[0]= vaddq_s8(vcntq_s8(veorq_s8(input.val[0],weight_cache_8.val[0])),output.val[0]);
                                    output.val[1]= vaddq_s8(vcntq_s8(veorq_s8(input.val[1],weight_cache_8.val[1])),output.val[1]);

                                    outputs[_k*out_height*out_width+_h*out_width+_w] += (int8_t) max(0,(int)(vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1])));
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}

int conv2d_3x3_vl512_s1(int8_t* inputs, int8_t* outputs, int8_t* filters, 
                        int height, int width,int num_filters, int num_channel,
                        int h_block, int w_block, int c_block, int k_block) {
    int filter_height = 3;
    int filter_width = 3;
    int padding = 2;
    int strides = 1;
    int out_height = height;
    int out_width = width;
    int i;
    int j;
    int input_h;
    int input_w;
    int prefetch_stride;
    int8_t* inputs_base;
    char filter_offset;

    int8x16x4_t input;
    int8x16x4_t output;
    int8x16x4_t weight;
    
    int8x16x4_t weight_cache_0;
    int8x16x4_t weight_cache_1;
    int8x16x4_t weight_cache_2;
    int8x16x4_t weight_cache_3;
    int8x16x4_t weight_cache_4;

    for (int c = 0; c < num_channel; c += c_block) {
        for (int _c = c; _c < min(c+c_block,num_channel); _c += 512) {
            for (int k = 0; k < num_filters; k += k_block) {
                for (int _k = k; _k < min(k+k_block, num_filters); _k++) {
                    weight_cache_0 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64*0);
                    weight_cache_1 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64*1);
                    weight_cache_2 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64*2);
                    weight_cache_3 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64*3);
                    weight_cache_4 = vld1q_s8_x4((const int8_t*)filters+_k*num_channel*filter_height*filter_width+_c/8*filter_height*filter_width+64*4);

                    for (int h = 0; h < out_height; h += h_block) {
                        for (int w = 0; w < out_width; w += w_block) {
                            for (int _h = h; _h < min(h+h_block,out_height); _h++) {
                                for (int _w = w; _w < min(w+w_block,out_width); _w++) {
                                    // __builtin_prefetch(inputs + _c * height * width + ((_h ) * width + _w) * 16, 0, 3);
                                    output.val[0] = vdupq_n_s8(0);
                                    output.val[1] = vdupq_n_s8(0);
                                    output.val[2] = vdupq_n_s8(0);
                                    output.val[3] = vdupq_n_s8(0);
                                    
                                    i = 0;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_0.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_0.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_0.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_0.val[3],output.val[3]);
                                    
                                    i = 0;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_1.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_1.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_1.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_1.val[3],output.val[3]);

                                    i = 0;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_2.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_2.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_2.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_2.val[3],output.val[3]);

                                    i = 1;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_3.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_3.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_3.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_3.val[3],output.val[3]);                                   
                                        
                                    i = 1;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight_cache_4.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight_cache_4.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight_cache_4.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight_cache_4.val[3],output.val[3]);

                                    i = 1;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);

                                    i = 2;
                                    j = 0;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);

                                    i = 2;
                                    j = 1;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c/8*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);
                                
                                    i = 2;
                                    j = 2;
                                    input_h = _h+i;
                                    input_w = _w+j;
                                    input = vld1q_s8_x4((const int8_t*)inputs+_c/8*height*width+(input_h*width+input_w)*64);
                                    weight = vld1q_s8_x4((const int8_t*)filters+_c/8*filter_height*filter_width+(i*filter_width+j)*64);
                                    output.val[0] = vmlaq_s8(input.val[0],weight.val[0],output.val[0]);
                                    output.val[1] = vmlaq_s8(input.val[1],weight.val[1],output.val[1]);
                                    output.val[2] = vmlaq_s8(input.val[2],weight.val[2],output.val[2]);
                                    output.val[3] = vmlaq_s8(input.val[3],weight.val[3],output.val[3]);
                                    outputs[_k*out_height*out_width+_h*out_width+_w] += (int8_t) max(0,(int)(vaddvq_s8(output.val[0])+vaddvq_s8(output.val[1])+vaddvq_s8(output.val[2])+vaddvq_s8(output.val[3])));
                                    
                                }
                            }
                        }
                    }
                } 
            }
        }
    }

    return 0;
}


#endif