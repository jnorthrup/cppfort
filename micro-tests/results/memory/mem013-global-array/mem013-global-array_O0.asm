
/Users/jim/work/cppfort/micro-tests/results/memory/mem013-global-array/mem013-global-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003f8 <__Z17test_global_arrayv>:
1000003f8: 90000028    	adrp	x8, 0x100004000 <_global_arr>
1000003fc: 91000108    	add	x8, x8, #0x0
100000400: b9400900    	ldr	w0, [x8, #0x8]
100000404: d65f03c0    	ret

0000000100000408 <_main>:
100000408: d10083ff    	sub	sp, sp, #0x20
10000040c: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000410: 910043fd    	add	x29, sp, #0x10
100000414: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000418: 97fffff8    	bl	0x1000003f8 <__Z17test_global_arrayv>
10000041c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000420: 910083ff    	add	sp, sp, #0x20
100000424: d65f03c0    	ret
