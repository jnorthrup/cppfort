
/Users/jim/work/cppfort/micro-tests/results/memory/mem044-pointer-to-member/mem044-pointer-to-member_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z22test_pointer_to_memberv>:
1000003b0: d10043ff    	sub	sp, sp, #0x10
1000003b4: 90000008    	adrp	x8, 0x100000000
1000003b8: 91100108    	add	x8, x8, #0x400
1000003bc: f9400109    	ldr	x9, [x8]
1000003c0: 910023e8    	add	x8, sp, #0x8
1000003c4: f90007e9    	str	x9, [sp, #0x8]
1000003c8: f90003ff    	str	xzr, [sp]
1000003cc: f94003e9    	ldr	x9, [sp]
1000003d0: b8696900    	ldr	w0, [x8, x9]
1000003d4: 910043ff    	add	sp, sp, #0x10
1000003d8: d65f03c0    	ret

00000001000003dc <_main>:
1000003dc: d10083ff    	sub	sp, sp, #0x20
1000003e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e4: 910043fd    	add	x29, sp, #0x10
1000003e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ec: 97fffff1    	bl	0x1000003b0 <__Z22test_pointer_to_memberv>
1000003f0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f4: 910083ff    	add	sp, sp, #0x20
1000003f8: d65f03c0    	ret
