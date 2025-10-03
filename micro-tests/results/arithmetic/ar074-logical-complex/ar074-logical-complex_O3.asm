
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar074-logical-complex/ar074-logical-complex_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_logical_complexbbb>:
100000360: 0a010008    	and	w8, w0, w1
100000364: 52000049    	eor	w9, w2, #0x1
100000368: 2a090100    	orr	w0, w8, w9
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800020    	mov	w0, #0x1                ; =1
100000374: d65f03c0    	ret
