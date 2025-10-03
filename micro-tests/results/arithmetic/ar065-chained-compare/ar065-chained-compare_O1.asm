
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar065-chained-compare/ar065-chained-compare_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_chained_compareiii>:
100000360: 6b01001f    	cmp	w0, w1
100000364: 7a42b020    	ccmp	w1, w2, #0x0, lt
100000368: 1a9fa7e0    	cset	w0, lt
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
