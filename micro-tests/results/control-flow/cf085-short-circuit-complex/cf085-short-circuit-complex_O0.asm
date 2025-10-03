
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf085-short-circuit-complex/cf085-short-circuit-complex_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z6check1Ri>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: f90007e0    	str	x0, [sp, #0x8]
100000368: f94007ea    	ldr	x10, [sp, #0x8]
10000036c: b9400149    	ldr	w9, [x10]
100000370: 52800028    	mov	w8, #0x1                ; =1
100000374: 11000529    	add	w9, w9, #0x1
100000378: b9000149    	str	w9, [x10]
10000037c: 12000100    	and	w0, w8, #0x1
100000380: 910043ff    	add	sp, sp, #0x10
100000384: d65f03c0    	ret

0000000100000388 <__Z6check2Ri>:
100000388: d10043ff    	sub	sp, sp, #0x10
10000038c: f90007e0    	str	x0, [sp, #0x8]
100000390: f94007e9    	ldr	x9, [sp, #0x8]
100000394: b9400128    	ldr	w8, [x9]
100000398: 11002908    	add	w8, w8, #0xa
10000039c: b9000128    	str	w8, [x9]
1000003a0: 52800008    	mov	w8, #0x0                ; =0
1000003a4: 12000100    	and	w0, w8, #0x1
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <__Z6check3Ri>:
1000003b0: d10043ff    	sub	sp, sp, #0x10
1000003b4: f90007e0    	str	x0, [sp, #0x8]
1000003b8: f94007e9    	ldr	x9, [sp, #0x8]
1000003bc: b9400128    	ldr	w8, [x9]
1000003c0: 11019108    	add	w8, w8, #0x64
1000003c4: b9000128    	str	w8, [x9]
1000003c8: 52800028    	mov	w8, #0x1                ; =1
1000003cc: 12000100    	and	w0, w8, #0x1
1000003d0: 910043ff    	add	sp, sp, #0x10
1000003d4: d65f03c0    	ret

00000001000003d8 <__Z26test_complex_short_circuitv>:
1000003d8: d10083ff    	sub	sp, sp, #0x20
1000003dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e0: 910043fd    	add	x29, sp, #0x10
1000003e4: d10013a0    	sub	x0, x29, #0x4
1000003e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ec: 97ffffdd    	bl	0x100000360 <__Z6check1Ri>
1000003f0: 52800008    	mov	w8, #0x0                ; =0
1000003f4: b90007e8    	str	w8, [sp, #0x4]
1000003f8: 36000180    	tbz	w0, #0x0, 0x100000428 <__Z26test_complex_short_circuitv+0x50>
1000003fc: 14000001    	b	0x100000400 <__Z26test_complex_short_circuitv+0x28>
100000400: d10013a0    	sub	x0, x29, #0x4
100000404: 97ffffe1    	bl	0x100000388 <__Z6check2Ri>
100000408: 52800008    	mov	w8, #0x0                ; =0
10000040c: b90007e8    	str	w8, [sp, #0x4]
100000410: 360000c0    	tbz	w0, #0x0, 0x100000428 <__Z26test_complex_short_circuitv+0x50>
100000414: 14000001    	b	0x100000418 <__Z26test_complex_short_circuitv+0x40>
100000418: d10013a0    	sub	x0, x29, #0x4
10000041c: 97ffffe5    	bl	0x1000003b0 <__Z6check3Ri>
100000420: b90007e0    	str	w0, [sp, #0x4]
100000424: 14000001    	b	0x100000428 <__Z26test_complex_short_circuitv+0x50>
100000428: b94007e8    	ldr	w8, [sp, #0x4]
10000042c: 12000108    	and	w8, w8, #0x1
100000430: 381fb3a8    	sturb	w8, [x29, #-0x5]
100000434: b85fc3a0    	ldur	w0, [x29, #-0x4]
100000438: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000043c: 910083ff    	add	sp, sp, #0x20
100000440: d65f03c0    	ret

0000000100000444 <_main>:
100000444: d10083ff    	sub	sp, sp, #0x20
100000448: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000044c: 910043fd    	add	x29, sp, #0x10
100000450: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000454: 97ffffe1    	bl	0x1000003d8 <__Z26test_complex_short_circuitv>
100000458: a9417bfd    	ldp	x29, x30, [sp, #0x10]
10000045c: 910083ff    	add	sp, sp, #0x20
100000460: d65f03c0    	ret
