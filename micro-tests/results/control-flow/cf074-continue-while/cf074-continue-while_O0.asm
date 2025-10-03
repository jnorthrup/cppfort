
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf074-continue-while/cf074-continue-while_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_continue_whilev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z19test_continue_whilev+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71005108    	subs	w8, w8, #0x14
100000378: 5400024a    	b.ge	0x1000003c0 <__Z19test_continue_whilev+0x60>
10000037c: 14000001    	b	0x100000380 <__Z19test_continue_whilev+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 11000508    	add	w8, w8, #0x1
100000388: b9000be8    	str	w8, [sp, #0x8]
10000038c: b9400be8    	ldr	w8, [sp, #0x8]
100000390: 5280006a    	mov	w10, #0x3               ; =3
100000394: 1aca0d09    	sdiv	w9, w8, w10
100000398: 1b0a7d29    	mul	w9, w9, w10
10000039c: 6b090108    	subs	w8, w8, w9
1000003a0: 35000068    	cbnz	w8, 0x1000003ac <__Z19test_continue_whilev+0x4c>
1000003a4: 14000001    	b	0x1000003a8 <__Z19test_continue_whilev+0x48>
1000003a8: 17fffff2    	b	0x100000370 <__Z19test_continue_whilev+0x10>
1000003ac: b9400be9    	ldr	w9, [sp, #0x8]
1000003b0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003b4: 0b090108    	add	w8, w8, w9
1000003b8: b9000fe8    	str	w8, [sp, #0xc]
1000003bc: 17ffffed    	b	0x100000370 <__Z19test_continue_whilev+0x10>
1000003c0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c4: 910043ff    	add	sp, sp, #0x10
1000003c8: d65f03c0    	ret

00000001000003cc <_main>:
1000003cc: d10083ff    	sub	sp, sp, #0x20
1000003d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d4: 910043fd    	add	x29, sp, #0x10
1000003d8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003dc: 97ffffe1    	bl	0x100000360 <__Z19test_continue_whilev>
1000003e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e4: 910083ff    	add	sp, sp, #0x20
1000003e8: d65f03c0    	ret
