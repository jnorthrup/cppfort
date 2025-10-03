
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar068-clamp-function/ar068-clamp-function_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z10test_clampiii>:
100000360: 6b02001f    	cmp	w0, w2
100000364: 1a82b008    	csel	w8, w0, w2, lt
100000368: 6b01001f    	cmp	w0, w1
10000036c: 1a88b020    	csel	w0, w1, w8, lt
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800140    	mov	w0, #0xa                ; =10
100000378: d65f03c0    	ret
