
/Users/jim/work/cppfort/micro-tests/results/memory/mem103-std-array/mem103-std-array_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z14test_std_arrayv>:
1000003b0: d100c3ff    	sub	sp, sp, #0x30
1000003b4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000003b8: 910083fd    	add	x29, sp, #0x20
1000003bc: 90000008    	adrp	x8, 0x100000000
1000003c0: 9110c108    	add	x8, x8, #0x430
1000003c4: 3dc00100    	ldr	q0, [x8]
1000003c8: 910003e0    	mov	x0, sp
1000003cc: 3d8003e0    	str	q0, [sp]
1000003d0: b9401108    	ldr	w8, [x8, #0x10]
1000003d4: b90013e8    	str	w8, [sp, #0x10]
1000003d8: d2800041    	mov	x1, #0x2                ; =2
1000003dc: 94000005    	bl	0x1000003f0 <__ZNSt3__15arrayIiLm5EEixB8ne200100Em>
1000003e0: b9400000    	ldr	w0, [x0]
1000003e4: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000003e8: 9100c3ff    	add	sp, sp, #0x30
1000003ec: d65f03c0    	ret

00000001000003f0 <__ZNSt3__15arrayIiLm5EEixB8ne200100Em>:
1000003f0: d10043ff    	sub	sp, sp, #0x10
1000003f4: f90007e0    	str	x0, [sp, #0x8]
1000003f8: f90003e1    	str	x1, [sp]
1000003fc: f94007e8    	ldr	x8, [sp, #0x8]
100000400: f94003e9    	ldr	x9, [sp]
100000404: 8b090900    	add	x0, x8, x9, lsl #2
100000408: 910043ff    	add	sp, sp, #0x10
10000040c: d65f03c0    	ret

0000000100000410 <_main>:
100000410: d10083ff    	sub	sp, sp, #0x20
100000414: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000418: 910043fd    	add	x29, sp, #0x10
10000041c: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000420: 97ffffe4    	bl	0x1000003b0 <__Z14test_std_arrayv>
100000424: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000428: 910083ff    	add	sp, sp, #0x20
10000042c: d65f03c0    	ret
