
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf022-for-continue/cf022-for-continue_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_for_continuev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z17test_for_continuev+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400026a    	b.ge	0x1000003c4 <__Z17test_for_continuev+0x64>
10000037c: 14000001    	b	0x100000380 <__Z17test_for_continuev+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 5280004a    	mov	w10, #0x2               ; =2
100000388: 1aca0d09    	sdiv	w9, w8, w10
10000038c: 1b0a7d29    	mul	w9, w9, w10
100000390: 6b090108    	subs	w8, w8, w9
100000394: 35000068    	cbnz	w8, 0x1000003a0 <__Z17test_for_continuev+0x40>
100000398: 14000001    	b	0x10000039c <__Z17test_for_continuev+0x3c>
10000039c: 14000006    	b	0x1000003b4 <__Z17test_for_continuev+0x54>
1000003a0: b9400be9    	ldr	w9, [sp, #0x8]
1000003a4: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a8: 0b090108    	add	w8, w8, w9
1000003ac: b9000fe8    	str	w8, [sp, #0xc]
1000003b0: 14000001    	b	0x1000003b4 <__Z17test_for_continuev+0x54>
1000003b4: b9400be8    	ldr	w8, [sp, #0x8]
1000003b8: 11000508    	add	w8, w8, #0x1
1000003bc: b9000be8    	str	w8, [sp, #0x8]
1000003c0: 17ffffec    	b	0x100000370 <__Z17test_for_continuev+0x10>
1000003c4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c8: 910043ff    	add	sp, sp, #0x10
1000003cc: d65f03c0    	ret

00000001000003d0 <_main>:
1000003d0: d10083ff    	sub	sp, sp, #0x20
1000003d4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d8: 910043fd    	add	x29, sp, #0x10
1000003dc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e0: 97ffffe0    	bl	0x100000360 <__Z17test_for_continuev>
1000003e4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e8: 910083ff    	add	sp, sp, #0x20
1000003ec: d65f03c0    	ret
