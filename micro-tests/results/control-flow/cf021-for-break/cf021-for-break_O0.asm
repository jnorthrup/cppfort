
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf021-for-break/cf021-for-break_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_for_breakv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z14test_for_breakv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71019108    	subs	w8, w8, #0x64
100000378: 5400020a    	b.ge	0x1000003b8 <__Z14test_for_breakv+0x58>
10000037c: 14000001    	b	0x100000380 <__Z14test_for_breakv+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 71002908    	subs	w8, w8, #0xa
100000388: 5400006b    	b.lt	0x100000394 <__Z14test_for_breakv+0x34>
10000038c: 14000001    	b	0x100000390 <__Z14test_for_breakv+0x30>
100000390: 1400000a    	b	0x1000003b8 <__Z14test_for_breakv+0x58>
100000394: b9400be9    	ldr	w9, [sp, #0x8]
100000398: b9400fe8    	ldr	w8, [sp, #0xc]
10000039c: 0b090108    	add	w8, w8, w9
1000003a0: b9000fe8    	str	w8, [sp, #0xc]
1000003a4: 14000001    	b	0x1000003a8 <__Z14test_for_breakv+0x48>
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 11000508    	add	w8, w8, #0x1
1000003b0: b9000be8    	str	w8, [sp, #0x8]
1000003b4: 17ffffef    	b	0x100000370 <__Z14test_for_breakv+0x10>
1000003b8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003bc: 910043ff    	add	sp, sp, #0x10
1000003c0: d65f03c0    	ret

00000001000003c4 <_main>:
1000003c4: d10083ff    	sub	sp, sp, #0x20
1000003c8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003cc: 910043fd    	add	x29, sp, #0x10
1000003d0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d4: 97ffffe3    	bl	0x100000360 <__Z14test_for_breakv>
1000003d8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003dc: 910083ff    	add	sp, sp, #0x20
1000003e0: d65f03c0    	ret
