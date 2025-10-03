
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar051-bit-toggle/ar051-bit-toggle_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_bit_toggleii>:
100000360: 52800028    	mov	w8, #0x1                ; =1
100000364: 1ac12108    	lsl	w8, w8, w1
100000368: 4a000100    	eor	w0, w8, w0
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52800100    	mov	w0, #0x8                ; =8
100000374: d65f03c0    	ret
