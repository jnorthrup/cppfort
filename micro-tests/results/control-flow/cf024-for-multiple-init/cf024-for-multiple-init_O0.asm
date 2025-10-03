
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf024-for-multiple-init/cf024-for-multiple-init_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_for_multi_initv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 52800148    	mov	w8, #0xa                ; =10
100000370: b90007e8    	str	w8, [sp, #0x4]
100000374: 14000001    	b	0x100000378 <__Z19test_for_multi_initv+0x18>
100000378: b9400be8    	ldr	w8, [sp, #0x8]
10000037c: b94007e9    	ldr	w9, [sp, #0x4]
100000380: 6b090108    	subs	w8, w8, w9
100000384: 5400020a    	b.ge	0x1000003c4 <__Z19test_for_multi_initv+0x64>
100000388: 14000001    	b	0x10000038c <__Z19test_for_multi_initv+0x2c>
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: b94007e9    	ldr	w9, [sp, #0x4]
100000394: 0b090109    	add	w9, w8, w9
100000398: b9400fe8    	ldr	w8, [sp, #0xc]
10000039c: 0b090108    	add	w8, w8, w9
1000003a0: b9000fe8    	str	w8, [sp, #0xc]
1000003a4: 14000001    	b	0x1000003a8 <__Z19test_for_multi_initv+0x48>
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 11000508    	add	w8, w8, #0x1
1000003b0: b9000be8    	str	w8, [sp, #0x8]
1000003b4: b94007e8    	ldr	w8, [sp, #0x4]
1000003b8: 71000508    	subs	w8, w8, #0x1
1000003bc: b90007e8    	str	w8, [sp, #0x4]
1000003c0: 17ffffee    	b	0x100000378 <__Z19test_for_multi_initv+0x18>
1000003c4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c8: 910043ff    	add	sp, sp, #0x10
1000003cc: d65f03c0    	ret

00000001000003d0 <_main>:
1000003d0: d10083ff    	sub	sp, sp, #0x20
1000003d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d8: 910043fd    	add	x29, sp, #0x10
1000003dc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e0: 97ffffe0    	bl	0x100000360 <__Z19test_for_multi_initv>
1000003e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e8: 910083ff    	add	sp, sp, #0x20
1000003ec: d65f03c0    	ret
