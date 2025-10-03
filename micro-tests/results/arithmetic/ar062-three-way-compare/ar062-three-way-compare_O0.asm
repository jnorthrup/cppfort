
/Users/jim/work/cppfort/micro-tests/results/arithmetic/ar062-three-way-compare/ar062-three-way-compare_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_three_wayii>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b90007e1    	str	w1, [sp, #0x4]
10000036c: b9400bea    	ldr	w10, [sp, #0x8]
100000370: b94007eb    	ldr	w11, [sp, #0x4]
100000374: 52800028    	mov	w8, #0x1                ; =1
100000378: 6b0b0149    	subs	w9, w10, w11
10000037c: 5a9fa109    	csinv	w9, w8, wzr, ge
100000380: 52800008    	mov	w8, #0x0                ; =0
100000384: 6b0b014a    	subs	w10, w10, w11
100000388: 1a890108    	csel	w8, w8, w9, eq
10000038c: 39003fe8    	strb	w8, [sp, #0xf]
100000390: 39403fe0    	ldrb	w0, [sp, #0xf]
100000394: 910043ff    	add	sp, sp, #0x10
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: d10083ff    	sub	sp, sp, #0x20
1000003a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a4: 910043fd    	add	x29, sp, #0x10
1000003a8: 52800008    	mov	w8, #0x0                ; =0
1000003ac: b90007e8    	str	w8, [sp, #0x4]
1000003b0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003b4: 528000a0    	mov	w0, #0x5                ; =5
1000003b8: 52800061    	mov	w1, #0x3                ; =3
1000003bc: 97ffffe9    	bl	0x100000360 <__Z14test_three_wayii>
1000003c0: b94003e8    	ldr	w8, [sp]
1000003c4: 381fb3a0    	sturb	w0, [x29, #-0x5]
1000003c8: 385fb3a9    	ldurb	w9, [x29, #-0x5]
1000003cc: 381fa3a9    	sturb	w9, [x29, #-0x6]
1000003d0: 381f93a8    	sturb	w8, [x29, #-0x7]
1000003d4: 385fa3a8    	ldurb	w8, [x29, #-0x6]
1000003d8: aa0803e0    	mov	x0, x8
1000003dc: 94000007    	bl	0x1000003f8 <__ZNSt3__1gtB8ne200100ENS_15strong_orderingENS_20_CmpUnspecifiedParamE>
1000003e0: b94007e8    	ldr	w8, [sp, #0x4]
1000003e4: 72000009    	ands	w9, w0, #0x1
1000003e8: 1a9f0500    	csinc	w0, w8, wzr, eq
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret

00000001000003f8 <__ZNSt3__1gtB8ne200100ENS_15strong_orderingENS_20_CmpUnspecifiedParamE>:
1000003f8: d10043ff    	sub	sp, sp, #0x10
1000003fc: aa0003e8    	mov	x8, x0
100000400: 39003fe8    	strb	w8, [sp, #0xf]
100000404: 39c03fe8    	ldrsb	w8, [sp, #0xf]
100000408: 71000108    	subs	w8, w8, #0x0
10000040c: 1a9fd7e0    	cset	w0, gt
100000410: 910043ff    	add	sp, sp, #0x10
100000414: d65f03c0    	ret
