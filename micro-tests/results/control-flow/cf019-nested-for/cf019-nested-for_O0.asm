
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf019-nested-for/cf019-nested-for_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_nested_forv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z15test_nested_forv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71001508    	subs	w8, w8, #0x5
100000378: 5400030a    	b.ge	0x1000003d8 <__Z15test_nested_forv+0x78>
10000037c: 14000001    	b	0x100000380 <__Z15test_nested_forv+0x20>
100000380: b90007ff    	str	wzr, [sp, #0x4]
100000384: 14000001    	b	0x100000388 <__Z15test_nested_forv+0x28>
100000388: b94007e8    	ldr	w8, [sp, #0x4]
10000038c: 71001508    	subs	w8, w8, #0x5
100000390: 540001aa    	b.ge	0x1000003c4 <__Z15test_nested_forv+0x64>
100000394: 14000001    	b	0x100000398 <__Z15test_nested_forv+0x38>
100000398: b9400be8    	ldr	w8, [sp, #0x8]
10000039c: b94007e9    	ldr	w9, [sp, #0x4]
1000003a0: 1b097d09    	mul	w9, w8, w9
1000003a4: b9400fe8    	ldr	w8, [sp, #0xc]
1000003a8: 0b090108    	add	w8, w8, w9
1000003ac: b9000fe8    	str	w8, [sp, #0xc]
1000003b0: 14000001    	b	0x1000003b4 <__Z15test_nested_forv+0x54>
1000003b4: b94007e8    	ldr	w8, [sp, #0x4]
1000003b8: 11000508    	add	w8, w8, #0x1
1000003bc: b90007e8    	str	w8, [sp, #0x4]
1000003c0: 17fffff2    	b	0x100000388 <__Z15test_nested_forv+0x28>
1000003c4: 14000001    	b	0x1000003c8 <__Z15test_nested_forv+0x68>
1000003c8: b9400be8    	ldr	w8, [sp, #0x8]
1000003cc: 11000508    	add	w8, w8, #0x1
1000003d0: b9000be8    	str	w8, [sp, #0x8]
1000003d4: 17ffffe7    	b	0x100000370 <__Z15test_nested_forv+0x10>
1000003d8: b9400fe0    	ldr	w0, [sp, #0xc]
1000003dc: 910043ff    	add	sp, sp, #0x10
1000003e0: d65f03c0    	ret

00000001000003e4 <_main>:
1000003e4: d10083ff    	sub	sp, sp, #0x20
1000003e8: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003ec: 910043fd    	add	x29, sp, #0x10
1000003f0: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003f4: 97ffffdb    	bl	0x100000360 <__Z15test_nested_forv>
1000003f8: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003fc: 910083ff    	add	sp, sp, #0x20
100000400: d65f03c0    	ret
