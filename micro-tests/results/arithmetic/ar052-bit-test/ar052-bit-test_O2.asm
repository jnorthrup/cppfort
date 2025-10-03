
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar052-bit-test/ar052-bit-test_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z13test_bit_testii>:
100000360: 1ac12408    	lsr	w8, w0, w1
100000364: 12000100    	and	w0, w8, #0x1
100000368: d65f03c0    	ret

000000010000036c <_main>:
10000036c: 52800020    	mov	w0, #0x1                ; =1
100000370: d65f03c0    	ret
