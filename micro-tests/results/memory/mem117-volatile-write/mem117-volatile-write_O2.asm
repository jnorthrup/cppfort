
/Users/jim/work/cppfort/micro-tests/results/memory/mem117-volatile-write/mem117-volatile-write_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z19test_volatile_writei>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_global_volatile>
1000003fc: b9000100    	str	w0, [x8]
100000400: d65f03c0    	ret

0000000100000404 <_main>:
100000404: 52800548    	mov	w8, #0x2a               ; =42
100000408: 90000029    	adrp	x9, 0x100004000 <_global_volatile>
10000040c: b9000128    	str	w8, [x9]
100000410: b9400120    	ldr	w0, [x9]
100000414: d65f03c0    	ret
