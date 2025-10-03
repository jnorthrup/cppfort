
/Users/jim/work/cppfort/micro-tests/results/memory/mem105-std-array-size/mem105-std-array-size_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z19test_std_array_sizev>:
1000003b0: d100c3ff    	sub	sp, sp, #0x30
1000003b4: a9027bfd    	stp	x29, x30, [sp, #0x20]
1000003b8: 910083fd    	add	x29, sp, #0x20
1000003bc: 90000008    	adrp	x8, 0x100000000
1000003c0: 91107108    	add	x8, x8, #0x41c
1000003c4: 3dc00100    	ldr	q0, [x8]
1000003c8: 910003e0    	mov	x0, sp
1000003cc: 3d8003e0    	str	q0, [sp]
1000003d0: b9401108    	ldr	w8, [x8, #0x10]
1000003d4: b90013e8    	str	w8, [sp, #0x10]
1000003d8: 94000004    	bl	0x1000003e8 <__ZNKSt3__15arrayIiLm5EE4sizeB8ne200100Ev>
1000003dc: a9427bfd    	ldp	x29, x30, [sp, #0x20]
1000003e0: 9100c3ff    	add	sp, sp, #0x30
1000003e4: d65f03c0    	ret

00000001000003e8 <__ZNKSt3__15arrayIiLm5EE4sizeB8ne200100Ev>:
1000003e8: d10043ff    	sub	sp, sp, #0x10
1000003ec: f90007e0    	str	x0, [sp, #0x8]
1000003f0: d28000a0    	mov	x0, #0x5                ; =5
1000003f4: 910043ff    	add	sp, sp, #0x10
1000003f8: d65f03c0    	ret

00000001000003fc <_main>:
1000003fc: d10083ff    	sub	sp, sp, #0x20
100000400: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000404: 910043fd    	add	x29, sp, #0x10
100000408: b81fc3bf    	stur	wzr, [x29, #-0x4]
10000040c: 97ffffe9    	bl	0x1000003b0 <__Z19test_std_array_sizev>
100000410: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000414: 910083ff    	add	sp, sp, #0x20
100000418: d65f03c0    	ret
