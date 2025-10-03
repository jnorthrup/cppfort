
/Users/jim/work/cppfort/micro-tests/results/memory/mem117-volatile-write/mem117-volatile-write_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z19test_volatile_writei>:
1000003f8: d10043ff    	sub	sp, sp, #0x10
1000003fc: b9000fe0    	str	w0, [sp, #0xc]
100000400: b9400fe8    	ldr	w8, [sp, #0xc]
100000404: 90000029    	adrp	x9, 0x100004000 <_global_volatile>
100000408: b9000128    	str	w8, [x9]
10000040c: 910043ff    	add	sp, sp, #0x10
100000410: d65f03c0    	ret

0000000100000414 <_main>:
100000414: d10083ff    	sub	sp, sp, #0x20
100000418: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000041c: 910043fd    	add	x29, sp, #0x10
100000420: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000424: 52800540    	mov	w0, #0x2a               ; =42
100000428: 97fffff4    	bl	0x1000003f8 <__Z19test_volatile_writei>
10000042c: 90000028    	adrp	x8, 0x100004000 <_global_volatile>
100000430: b9400100    	ldr	w0, [x8]
100000434: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000438: 910083ff    	add	sp, sp, #0x20
10000043c: d65f03c0    	ret
