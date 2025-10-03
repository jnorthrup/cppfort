
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf044-do-while-continue/cf044-do-while-continue_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z22test_do_while_continuev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z22test_do_while_continuev+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 11000508    	add	w8, w8, #0x1
100000378: b9000be8    	str	w8, [sp, #0x8]
10000037c: b9400be8    	ldr	w8, [sp, #0x8]
100000380: 5280004a    	mov	w10, #0x2               ; =2
100000384: 1aca0d09    	sdiv	w9, w8, w10
100000388: 1b0a7d29    	mul	w9, w9, w10
10000038c: 6b090108    	subs	w8, w8, w9
100000390: 35000068    	cbnz	w8, 0x10000039c <__Z22test_do_while_continuev+0x3c>
100000394: 14000001    	b	0x100000398 <__Z22test_do_while_continuev+0x38>
100000398: 14000006    	b	0x1000003b0 <__Z22test_do_while_continuev+0x50>
10000039c: b9400be9    	ldr	w9, [sp, #0x8]
1000003a0: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a4: 0b090108    	add	w8, w8, w9
1000003a8: b9000fe8    	str	w8, [sp, #0xc]
1000003ac: 14000001    	b	0x1000003b0 <__Z22test_do_while_continuev+0x50>
1000003b0: b9400be8    	ldr	w8, [sp, #0x8]
1000003b4: 71002908    	subs	w8, w8, #0xa
1000003b8: 54fffdcb    	b.lt	0x100000370 <__Z22test_do_while_continuev+0x10>
1000003bc: 14000001    	b	0x1000003c0 <__Z22test_do_while_continuev+0x60>
1000003c0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c4: 910043ff    	add	sp, sp, #0x10
1000003c8: d65f03c0    	ret

00000001000003cc <_main>:
1000003cc: d10083ff    	sub	sp, sp, #0x20
1000003d0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d4: 910043fd    	add	x29, sp, #0x10
1000003d8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003dc: 97ffffe1    	bl	0x100000360 <__Z22test_do_while_continuev>
1000003e0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e4: 910083ff    	add	sp, sp, #0x20
1000003e8: d65f03c0    	ret
