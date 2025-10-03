
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf072-continue-pattern/cf072-continue-pattern_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_continue_patternv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z21test_continue_patternv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71005108    	subs	w8, w8, #0x14
100000378: 540003aa    	b.ge	0x1000003ec <__Z21test_continue_patternv+0x8c>
10000037c: 14000001    	b	0x100000380 <__Z21test_continue_patternv+0x20>
100000380: b9400be8    	ldr	w8, [sp, #0x8]
100000384: 71001508    	subs	w8, w8, #0x5
100000388: 5400006a    	b.ge	0x100000394 <__Z21test_continue_patternv+0x34>
10000038c: 14000001    	b	0x100000390 <__Z21test_continue_patternv+0x30>
100000390: 14000013    	b	0x1000003dc <__Z21test_continue_patternv+0x7c>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 71003d08    	subs	w8, w8, #0xf
10000039c: 5400006d    	b.le	0x1000003a8 <__Z21test_continue_patternv+0x48>
1000003a0: 14000001    	b	0x1000003a4 <__Z21test_continue_patternv+0x44>
1000003a4: 1400000e    	b	0x1000003dc <__Z21test_continue_patternv+0x7c>
1000003a8: b9400be8    	ldr	w8, [sp, #0x8]
1000003ac: 5280004a    	mov	w10, #0x2               ; =2
1000003b0: 1aca0d09    	sdiv	w9, w8, w10
1000003b4: 1b0a7d29    	mul	w9, w9, w10
1000003b8: 6b090108    	subs	w8, w8, w9
1000003bc: 35000068    	cbnz	w8, 0x1000003c8 <__Z21test_continue_patternv+0x68>
1000003c0: 14000001    	b	0x1000003c4 <__Z21test_continue_patternv+0x64>
1000003c4: 14000006    	b	0x1000003dc <__Z21test_continue_patternv+0x7c>
1000003c8: b9400be9    	ldr	w9, [sp, #0x8]
1000003cc: b9400fe8    	ldr	w8, [sp, #0xc]
1000003d0: 0b090108    	add	w8, w8, w9
1000003d4: b9000fe8    	str	w8, [sp, #0xc]
1000003d8: 14000001    	b	0x1000003dc <__Z21test_continue_patternv+0x7c>
1000003dc: b9400be8    	ldr	w8, [sp, #0x8]
1000003e0: 11000508    	add	w8, w8, #0x1
1000003e4: b9000be8    	str	w8, [sp, #0x8]
1000003e8: 17ffffe2    	b	0x100000370 <__Z21test_continue_patternv+0x10>
1000003ec: b9400fe0    	ldr	w0, [sp, #0xc]
1000003f0: 910043ff    	add	sp, sp, #0x10
1000003f4: d65f03c0    	ret

00000001000003f8 <_main>:
1000003f8: d10083ff    	sub	sp, sp, #0x20
1000003fc: a9017bfd    	stp	x29, x30, [sp, #0x10]
100000400: 910043fd    	add	x29, sp, #0x10
100000404: b81fc3bf    	stur	wzr, [x29, #-0x4]
100000408: 97ffffd6    	bl	0x100000360 <__Z21test_continue_patternv>
10000040c: a9417bfd    	ldp	x29, x30, [sp, #0x10]
100000410: 910083ff    	add	sp, sp, #0x20
100000414: d65f03c0    	ret
