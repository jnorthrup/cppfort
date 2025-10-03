
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf039-while-empty-body/cf039-while-empty-body_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_while_emptyv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: 14000001    	b	0x10000036c <__Z16test_while_emptyv+0xc>
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 11000509    	add	w9, w8, #0x1
100000374: b9000fe9    	str	w9, [sp, #0xc]
100000378: 71002908    	subs	w8, w8, #0xa
10000037c: 5400006a    	b.ge	0x100000388 <__Z16test_while_emptyv+0x28>
100000380: 14000001    	b	0x100000384 <__Z16test_while_emptyv+0x24>
100000384: 17fffffa    	b	0x10000036c <__Z16test_while_emptyv+0xc>
100000388: b9400fe0    	ldr	w0, [sp, #0xc]
10000038c: 910043ff    	add	sp, sp, #0x10
100000390: d65f03c0    	ret

0000000100000394 <_main>:
100000394: d10083ff    	sub	sp, sp, #0x20
100000398: a9017bfd    	stp	x29, x30, [sp, #0x10]
10000039c: 910043fd    	add	x29, sp, #0x10
1000003a0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003a4: 97ffffef    	bl	0x100000360 <__Z16test_while_emptyv>
1000003a8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ac: 910083ff    	add	sp, sp, #0x20
1000003b0: d65f03c0    	ret
