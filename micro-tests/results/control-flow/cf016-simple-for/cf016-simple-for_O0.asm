
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf016-simple-for/cf016-simple-for_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_simple_forv>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000fff    	str	wzr, [sp, #0xc]
100000368: b9000bff    	str	wzr, [sp, #0x8]
10000036c: 14000001    	b	0x100000370 <__Z15test_simple_forv+0x10>
100000370: b9400be8    	ldr	w8, [sp, #0x8]
100000374: 71002908    	subs	w8, w8, #0xa
100000378: 5400016a    	b.ge	0x1000003a4 <__Z15test_simple_forv+0x44>
10000037c: 14000001    	b	0x100000380 <__Z15test_simple_forv+0x20>
100000380: b9400be9    	ldr	w9, [sp, #0x8]
100000384: b9400fe8    	ldr	w8, [sp, #0xc]
100000388: 0b090108    	add	w8, w8, w9
10000038c: b9000fe8    	str	w8, [sp, #0xc]
100000390: 14000001    	b	0x100000394 <__Z15test_simple_forv+0x34>
100000394: b9400be8    	ldr	w8, [sp, #0x8]
100000398: 11000508    	add	w8, w8, #0x1
10000039c: b9000be8    	str	w8, [sp, #0x8]
1000003a0: 17fffff4    	b	0x100000370 <__Z15test_simple_forv+0x10>
1000003a4: b9400fe0    	ldr	w0, [sp, #0xc]
1000003a8: 910043ff    	add	sp, sp, #0x10
1000003ac: d65f03c0    	ret

00000001000003b0 <_main>:
1000003b0: d10083ff    	sub	sp, sp, #0x20
1000003b4: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003b8: 910043fd    	add	x29, sp, #0x10
1000003bc: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003c0: 97ffffe8    	bl	0x100000360 <__Z15test_simple_forv>
1000003c4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003c8: 910083ff    	add	sp, sp, #0x20
1000003cc: d65f03c0    	ret
