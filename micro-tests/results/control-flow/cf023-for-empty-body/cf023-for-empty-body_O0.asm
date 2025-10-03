
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf023-for-empty-body/cf023-for-empty-body_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z14test_for_emptyv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: 14000001    	b	0x10000036c <__Z14test_for_emptyv+0xc>
10000036c: b9400fe8    	ldr	w8, [sp, #0xc]
100000370: 71002908    	subs	w8, w8, #0xa
100000374: 540000ea    	b.ge	0x100000390 <__Z14test_for_emptyv+0x30>
100000378: 14000001    	b	0x10000037c <__Z14test_for_emptyv+0x1c>
10000037c: 14000001    	b	0x100000380 <__Z14test_for_emptyv+0x20>
100000380: b9400fe8    	ldr	w8, [sp, #0xc]
100000384: 11000508    	add	w8, w8, #0x1
100000388: b9000fe8    	str	w8, [sp, #0xc]
10000038c: 17fffff8    	b	0x10000036c <__Z14test_for_emptyv+0xc>
100000390: b9400fe0    	ldr	w0, [sp, #0xc]
100000394: 910043ff    	add	sp, sp, #0x10
100000398: d65f03c0    	ret

000000010000039c <_main>:
10000039c: d10083ff    	sub	sp, sp, #0x20
1000003a0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003a4: 910043fd    	add	x29, sp, #0x10
1000003a8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ac: 97ffffed    	bl	0x100000360 <__Z14test_for_emptyv>
1000003b0: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003b4: 910083ff    	add	sp, sp, #0x20
1000003b8: d65f03c0    	ret
