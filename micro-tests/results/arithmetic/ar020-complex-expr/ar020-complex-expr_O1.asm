
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar020-complex-expr/ar020-complex-expr_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z12test_complexiiii>:
100000360: 0b000028    	add	w8, w1, w0
100000364: 4b030049    	sub	w9, w2, w3
100000368: 1b087d28    	mul	w8, w9, w8
10000036c: 0b487d08    	add	w8, w8, w8, lsr #31
100000370: 13017d00    	asr	w0, w8, #1
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800f00    	mov	w0, #0x78               ; =120
10000037c: d65f03c0    	ret
