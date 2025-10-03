
/Users/jim/work/cppfort/micro-tests/results/functions/fn072-lambda/fn072-lambda_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 90000008    	adrp	x8, 0x100000000
1000003c4: 39500108    	ldrb	w8, [x8, #0x400]
1000003c8: d10017a0    	sub	x0, x29, #0x5
1000003cc: 381fb3a8    	sturb	w8, [x29, #-0x5]
1000003d0: 52800901    	mov	w1, #0x48               ; =72
1000003d4: 94000004    	bl	0x1000003e4 <__ZZ4mainENK3$_0clEi>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret

00000001000003e4 <__ZZ4mainENK3$_0clEi>:
1000003e4: d10043ff    	sub	sp, sp, #0x10
1000003e8: f90007e0    	str	x0, [sp, #0x8]
1000003ec: b90007e1    	str	w1, [sp, #0x4]
1000003f0: b94007e8    	ldr	w8, [sp, #0x4]
1000003f4: 11012100    	add	w0, w8, #0x48
1000003f8: 910043ff    	add	sp, sp, #0x10
1000003fc: d65f03c0    	ret
