
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar050-bit-clear/ar050-bit-clear_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_bit_clearii>:
100000360: 52800028    	mov	w8, #0x1                ; =1
100000364: 1ac12108    	lsl	w8, w8, w1
100000368: 0a280000    	bic	w0, w0, w8
10000036c: d65f03c0    	ret

0000000100000370 <_main>:
100000370: 52801ee0    	mov	w0, #0xf7               ; =247
100000374: d65f03c0    	ret
