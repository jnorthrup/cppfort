
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar005-mod-int/ar005-mod-int_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z8test_modii>:
100000360: 34000081    	cbz	w1, 0x100000370 <__Z8test_modii+0x10>
100000364: 1ac10c08    	sdiv	w8, w0, w1
100000368: 1b018100    	msub	w0, w8, w1, w0
10000036c: d65f03c0    	ret
100000370: 52800000    	mov	w0, #0x0                ; =0
100000374: d65f03c0    	ret

0000000100000378 <_main>:
100000378: 52800040    	mov	w0, #0x2                ; =2
10000037c: d65f03c0    	ret
