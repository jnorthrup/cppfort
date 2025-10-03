
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf046-do-while-complex-condition/cf046-do-while-complex-condition_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_do_while_complexv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z21test_do_while_complexv+0x10>
100000370: b9400be9    	ldr	w9, [sp, #0x8]
100000374: b9400fe8    	ldr	w8, [sp, #0xc]
100000378: 0b090108    	add	w8, w8, w9
10000037c: b9000fe8    	str	w8, [sp, #0xc]
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 11000508    	add	w8, w8, #0x1
100000388: b9000be8    	str	w8, [sp, #0x8]
10000038c: 14000001    	b	0x100000390 <__Z21test_do_while_complexv+0x30>
100000390: b9400be9    	ldr	w9, [sp, #0x8]
100000394: 52800008    	mov	w8, #0x0                ; =0
100000398: 71002929    	subs	w9, w9, #0xa
10000039c: b90007e8    	str	w8, [sp, #0x4]
1000003a0: 540000ea    	b.ge	0x1000003bc <__Z21test_do_while_complexv+0x5c>
1000003a4: 14000001    	b	0x1000003a8 <__Z21test_do_while_complexv+0x48>
1000003a8: b9400fe8    	ldr	w8, [sp, #0xc]
1000003ac: 71005108    	subs	w8, w8, #0x14
1000003b0: 1a9fa7e8    	cset	w8, lt
1000003b4: b90007e8    	str	w8, [sp, #0x4]
1000003b8: 14000001    	b	0x1000003bc <__Z21test_do_while_complexv+0x5c>
1000003bc: b94007e8    	ldr	w8, [sp, #0x4]
1000003c0: 3707fd88    	tbnz	w8, #0x0, 0x100000370 <__Z21test_do_while_complexv+0x10>
1000003c4: 14000001    	b	0x1000003c8 <__Z21test_do_while_complexv+0x68>
1000003c8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003cc: 910043ff    	add	sp, sp, #0x10
1000003d0: d65f03c0    	ret

00000001000003d4 <_main>:
1000003d4: d10083ff    	sub	sp, sp, #0x20
1000003d8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003dc: 910043fd    	add	x29, sp, #0x10
1000003e0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003e4: 97ffffdf    	bl	0x100000360 <__Z21test_do_while_complexv>
1000003e8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003ec: 910083ff    	add	sp, sp, #0x20
1000003f0: d65f03c0    	ret
