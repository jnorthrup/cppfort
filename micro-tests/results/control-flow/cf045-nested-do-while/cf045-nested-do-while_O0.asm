
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf045-nested-do-while/cf045-nested-do-while_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z20test_nested_do_whilev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z20test_nested_do_whilev+0x10>
100000370: b90007ff    	str	wzr, [sp, #0x4]
100000374: 14000001    	b	0x100000378 <__Z20test_nested_do_whilev+0x18>
100000378: b9400be8    	ldr	w8, [sp, #0x8]
10000037c: b94007e9    	ldr	w9, [sp, #0x4]
100000380: 1b097d09    	mul	w9, w8, w9
100000384: b9400fe8    	ldr	w8, [sp, #0xc]
100000388: 0b090108    	add	w8, w8, w9
10000038c: b9000fe8    	str	w8, [sp, #0xc]
100000390: b94007e8    	ldr	w8, [sp, #0x4]
100000394: 11000508    	add	w8, w8, #0x1
100000398: b90007e8    	str	w8, [sp, #0x4]
10000039c: 14000001    	b	0x1000003a0 <__Z20test_nested_do_whilev+0x40>
1000003a0: b94007e8    	ldr	w8, [sp, #0x4]
1000003a4: 71001508    	subs	w8, w8, #0x5
1000003a8: 54fffe8b    	b.lt	0x100000378 <__Z20test_nested_do_whilev+0x18>
1000003ac: 14000001    	b	0x1000003b0 <__Z20test_nested_do_whilev+0x50>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: 11000508    	add	w8, w8, #0x1
1000003b8: b9000be8    	str	w8, [sp, #0x8]
1000003bc: 14000001    	b	0x1000003c0 <__Z20test_nested_do_whilev+0x60>
1000003c0: b9400be8    	ldr	w8, [sp, #0x8]
1000003c4: 71001508    	subs	w8, w8, #0x5
1000003c8: 54fffd4b    	b.lt	0x100000370 <__Z20test_nested_do_whilev+0x10>
1000003cc: 14000001    	b	0x1000003d0 <__Z20test_nested_do_whilev+0x70>
1000003d0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003d4: 910043ff    	add	sp, sp, #0x10
1000003d8: d65f03c0    	ret

00000001000003dc <_main>:
1000003dc: d10083ff    	sub	sp, sp, #0x20
1000003e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e4: 910043fd    	add	x29, sp, #0x10
1000003e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ec: 97ffffdd    	bl	0x100000360 <__Z20test_nested_do_whilev>
1000003f0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f4: 910083ff    	add	sp, sp, #0x20
1000003f8: d65f03c0    	ret
