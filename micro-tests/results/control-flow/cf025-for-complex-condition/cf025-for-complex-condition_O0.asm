
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf025-for-complex-condition/cf025-for-complex-condition_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_for_complex_condv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z21test_for_complex_condv+0x10>
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: 52800008    	mov	w8, #0x0                ; =0
100000378: 71002929    	subs	w9, w9, #0xa
10000037c: b90007e8    	str	w8, [sp, #0x4]
100000380: 540000ea    	b.ge	0x10000039c <__Z21test_for_complex_condv+0x3c>
100000384: 14000001    	b	0x100000388 <__Z21test_for_complex_condv+0x28>
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: 71005108    	subs	w8, w8, #0x14
100000390: 1a9fa7e8    	cset	w8, lt
100000394: b90007e8    	str	w8, [sp, #0x4]
100000398: 14000001    	b	0x10000039c <__Z21test_for_complex_condv+0x3c>
10000039c: b94007e8    	ldr	w8, [sp, #0x4]
1000003a0: 36000168    	tbz	w8, #0x0, 0x1000003cc <__Z21test_for_complex_condv+0x6c>
1000003a4: 14000001    	b	0x1000003a8 <__Z21test_for_complex_condv+0x48>
1000003a8: b9400be9    	ldr	w9, [sp, #0x8]
1000003ac: b9400fe8    	ldr	w8, [sp, #0xc]
1000003b0: 0b090108    	add	w8, w8, w9
1000003b4: b9000fe8    	str	w8, [sp, #0xc]
1000003b8: 14000001    	b	0x1000003bc <__Z21test_for_complex_condv+0x5c>
1000003bc: b9400be8    	ldr	w8, [sp, #0x8]
1000003c0: 11000508    	add	w8, w8, #0x1
1000003c4: b9000be8    	str	w8, [sp, #0x8]
1000003c8: 17ffffea    	b	0x100000370 <__Z21test_for_complex_condv+0x10>
1000003cc: b9400fe0    	ldr	w0, [sp, #0xc]
1000003d0: 910043ff    	add	sp, sp, #0x10
1000003d4: d65f03c0    	ret

00000001000003d8 <_main>:
1000003d8: d10083ff    	sub	sp, sp, #0x20
1000003dc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e0: 910043fd    	add	x29, sp, #0x10
1000003e4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e8: 97ffffde    	bl	0x100000360 <__Z21test_for_complex_condv>
1000003ec: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f0: 910083ff    	add	sp, sp, #0x20
1000003f4: d65f03c0    	ret
