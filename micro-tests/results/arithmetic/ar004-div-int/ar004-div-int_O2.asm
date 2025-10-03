
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar004-div-int/ar004-div-int_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_divii>:
100000360: 34000061    	cbz	w1, 0x10000036c <__Z8test_divii+0xc>
100000364: 1ac10c00    	sdiv	w0, w0, w1
100000368: d65f03c0    	ret
10000036c: 52800000    	mov	w0, #0x0                ; =0
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 528000a0    	mov	w0, #0x5                ; =5
100000378: d65f03c0    	ret
