
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar035-float-precision/ar035-float-precision_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_precisionv>:
100000360: d10083ff    	sub	sp, sp, #0x20
100000364: b201e7e8    	mov	x8, #-0x6666666666666667 ; =-7378697629483820647
100000368: f2933348    	movk	x8, #0x999a
10000036c: f2e7f728    	movk	x8, #0x3fb9, lsl #48
100000370: 9e670100    	fmov	d0, x8
100000374: fd000fe0    	str	d0, [sp, #0x18]
100000378: b201e7e8    	mov	x8, #-0x6666666666666667 ; =-7378697629483820647
10000037c: f2933348    	movk	x8, #0x999a
100000380: f2e7f928    	movk	x8, #0x3fc9, lsl #48
100000384: 9e670100    	fmov	d0, x8
100000388: fd000be0    	str	d0, [sp, #0x10]
10000038c: b200e7e8    	mov	x8, #0x3333333333333333 ; =3689348814741910323
100000390: f2e7fa68    	movk	x8, #0x3fd3, lsl #48
100000394: 9e670100    	fmov	d0, x8
100000398: fd0007e0    	str	d0, [sp, #0x8]
10000039c: fd400fe0    	ldr	d0, [sp, #0x18]
1000003a0: fd400be1    	ldr	d1, [sp, #0x10]
1000003a4: 1e612800    	fadd	d0, d0, d1
1000003a8: fd4007e1    	ldr	d1, [sp, #0x8]
1000003ac: 52800008    	mov	w8, #0x0                ; =0
1000003b0: 1e612000    	fcmp	d0, d1
1000003b4: 1a9f1508    	csinc	w8, w8, wzr, ne
1000003b8: 1e620100    	scvtf	d0, w8
1000003bc: 910083ff    	add	sp, sp, #0x20
1000003c0: d65f03c0    	ret

00000001000003c4 <_main>:
1000003c4: d10083ff    	sub	sp, sp, #0x20
1000003c8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003cc: 910043fd    	add	x29, sp, #0x10
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: 97ffffe3    	bl	0x100000360 <__Z14test_precisionv>
1000003d8: 1e780000    	fcvtzs	w0, d0
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret
