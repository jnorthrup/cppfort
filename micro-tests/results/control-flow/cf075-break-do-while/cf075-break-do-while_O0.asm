
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf075-break-do-while/cf075-break-do-while_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_break_do_whilev>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z19test_break_do_whilev+0x10>
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: 0b090108    	add	w8, w8, w9
10000037c: b9000fe8    	str	w8, [sp, #0xc]
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 11000508    	add	w8, w8, #0x1
100000388: b9000be8    	str	w8, [sp, #0x8]
10000038c: b9400fe8    	ldr	w8, [sp, #0xc]
100000390: 71005108    	subs	w8, w8, #0x14
100000394: 5400006d    	b.le	0x1000003a0 <__Z19test_break_do_whilev+0x40>
100000398: 14000001    	b	0x10000039c <__Z19test_break_do_whilev+0x3c>
10000039c: 14000006    	b	0x1000003b4 <__Z19test_break_do_whilev+0x54>
1000003a0: 14000001    	b	0x1000003a4 <__Z19test_break_do_whilev+0x44>
1000003a4: b9400be8    	ldr	w8, [sp, #0x8]
1000003a8: 71019108    	subs	w8, w8, #0x64
1000003ac: 54fffe2b    	b.lt	0x100000370 <__Z19test_break_do_whilev+0x10>
1000003b0: 14000001    	b	0x1000003b4 <__Z19test_break_do_whilev+0x54>
1000003b4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b8: 910043ff    	add	sp, sp, #0x10
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: d10083ff    	sub	sp, sp, #0x20
1000003c4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c8: 910043fd    	add	x29, sp, #0x10
1000003cc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d0: 97ffffe4    	bl	0x100000360 <__Z19test_break_do_whilev>
1000003d4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d8: 910083ff    	add	sp, sp, #0x20
1000003dc: d65f03c0    	ret
