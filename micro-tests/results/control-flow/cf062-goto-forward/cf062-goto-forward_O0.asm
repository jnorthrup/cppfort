
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf062-goto-forward/cf062-goto-forward_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_goto_forwardv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z17test_goto_forwardv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400020a    	b.ge	0x1000003b8 <__Z17test_goto_forwardv+0x58>
10000037c: 14000001    	b	0x100000380 <__Z17test_goto_forwardv+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 71001508    	subs	w8, w8, #0x5
100000388: 54000061    	b.ne	0x100000394 <__Z17test_goto_forwardv+0x34>
10000038c: 14000001    	b	0x100000390 <__Z17test_goto_forwardv+0x30>
100000390: 1400000b    	b	0x1000003bc <__Z17test_goto_forwardv+0x5c>
100000394: b9400be9    	ldr	w9, [sp, #0x8]
100000398: b9400fe8    	ldr	w8, [sp, #0xc]
10000039c: 0b090108    	add	w8, w8, w9
1000003a0: b9000fe8    	str	w8, [sp, #0xc]
1000003a4: 14000001    	b	0x1000003a8 <__Z17test_goto_forwardv+0x48>
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 11000508    	add	w8, w8, #0x1
1000003b0: b9000be8    	str	w8, [sp, #0x8]
1000003b4: 17ffffef    	b	0x100000370 <__Z17test_goto_forwardv+0x10>
1000003b8: 14000001    	b	0x1000003bc <__Z17test_goto_forwardv+0x5c>
1000003bc: b9400fe0    	ldr	w0, [sp, #0xc]
1000003c0: 910043ff    	add	sp, sp, #0x10
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: d10083ff    	sub	sp, sp, #0x20
1000003cc: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003d0: 910043fd    	add	x29, sp, #0x10
1000003d4: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003d8: 97ffffe2    	bl	0x100000360 <__Z17test_goto_forwardv>
1000003dc: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003e0: 910083ff    	add	sp, sp, #0x20
1000003e4: d65f03c0    	ret
