
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar055-power-of-two/ar055-power-of-two_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_is_power_of_twoj>:
100000360: 51000408    	sub	w8, w0, #0x1
100000364: 6a08001f    	tst	w0, w8
100000368: 1a9f17e8    	cset	w8, eq
10000036c: 7100001f    	cmp	w0, #0x0
100000370: 1a8803e0    	csel	w0, wzr, w8, eq
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800020    	mov	w0, #0x1                ; =1
10000037c: d65f03c0    	ret
