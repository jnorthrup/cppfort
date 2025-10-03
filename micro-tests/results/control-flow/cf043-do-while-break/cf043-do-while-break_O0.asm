
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf043-do-while-break/cf043-do-while-break_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_do_while_breakv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z19test_do_while_breakv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400006b    	b.lt	0x100000384 <__Z19test_do_while_breakv+0x24>
10000037c: 14000001    	b	0x100000380 <__Z19test_do_while_breakv+0x20>
100000380: 1400000c    	b	0x1000003b0 <__Z19test_do_while_breakv+0x50>
100000384: b9400be9    	ldr	w9, [sp, #0x8]
100000388: b9400fe8    	ldr	w8, [sp, #0xc]
10000038c: 0b090108    	add	w8, w8, w9
100000390: b9000fe8    	str	w8, [sp, #0xc]
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 11000508    	add	w8, w8, #0x1
10000039c: b9000be8    	str	w8, [sp, #0x8]
1000003a0: 14000001    	b	0x1000003a4 <__Z19test_do_while_breakv+0x44>
1000003a4: 52800028    	mov	w8, #0x1                ; =1
1000003a8: 3707fe48    	tbnz	w8, #0x0, 0x100000370 <__Z19test_do_while_breakv+0x10>
1000003ac: 14000001    	b	0x1000003b0 <__Z19test_do_while_breakv+0x50>
1000003b0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003b4: 910043ff    	add	sp, sp, #0x10
1000003b8: d65f03c0    	ret

00000001000003bc <_main>:
1000003bc: d10083ff    	sub	sp, sp, #0x20
1000003c0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003c4: 910043fd    	add	x29, sp, #0x10
1000003c8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003cc: 97ffffe5    	bl	0x100000360 <__Z19test_do_while_breakv>
1000003d0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003d4: 910083ff    	add	sp, sp, #0x20
1000003d8: d65f03c0    	ret
