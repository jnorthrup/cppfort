
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf068-goto-error-handling/cf068-goto-error-handling_O0.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_goto_errori>:
100000360: d10043ff    	sub	sp, sp, #0x10
100000364: b9000be0    	str	w0, [sp, #0x8]
100000368: b9400be8    	ldr	w8, [sp, #0x8]
10000036c: 36f80068    	tbz	w8, #0x1f, 0x100000378 <__Z15test_goto_errori+0x18>
100000370: 14000001    	b	0x100000374 <__Z15test_goto_errori+0x14>
100000374: 1400000e    	b	0x1000003ac <__Z15test_goto_errori+0x4c>
100000378: b9400be8    	ldr	w8, [sp, #0x8]
10000037c: 35000068    	cbnz	w8, 0x100000388 <__Z15test_goto_errori+0x28>
100000380: 14000001    	b	0x100000384 <__Z15test_goto_errori+0x24>
100000384: 1400000d    	b	0x1000003b8 <__Z15test_goto_errori+0x58>
100000388: b9400be8    	ldr	w8, [sp, #0x8]
10000038c: 71019108    	subs	w8, w8, #0x64
100000390: 5400006d    	b.le	0x10000039c <__Z15test_goto_errori+0x3c>
100000394: 14000001    	b	0x100000398 <__Z15test_goto_errori+0x38>
100000398: 1400000b    	b	0x1000003c4 <__Z15test_goto_errori+0x64>
10000039c: b9400be8    	ldr	w8, [sp, #0x8]
1000003a0: 531f7908    	lsl	w8, w8, #1
1000003a4: b9000fe8    	str	w8, [sp, #0xc]
1000003a8: 1400000a    	b	0x1000003d0 <__Z15test_goto_errori+0x70>
1000003ac: 12800008    	mov	w8, #-0x1               ; =-1
1000003b0: b9000fe8    	str	w8, [sp, #0xc]
1000003b4: 14000007    	b	0x1000003d0 <__Z15test_goto_errori+0x70>
1000003b8: 12800028    	mov	w8, #-0x2               ; =-2
1000003bc: b9000fe8    	str	w8, [sp, #0xc]
1000003c0: 14000004    	b	0x1000003d0 <__Z15test_goto_errori+0x70>
1000003c4: 12800048    	mov	w8, #-0x3               ; =-3
1000003c8: b9000fe8    	str	w8, [sp, #0xc]
1000003cc: 14000001    	b	0x1000003d0 <__Z15test_goto_errori+0x70>
1000003d0: b9400fe0    	ldr	w0, [sp, #0xc]
1000003d4: 910043ff    	add	sp, sp, #0x10
1000003d8: d65f03c0    	ret

00000001000003dc <_main>:
1000003dc: d10083ff    	sub	sp, sp, #0x20
1000003e0: a9017bfd    	stp	x29, x30, [sp, #0x10]
1000003e4: 910043fd    	add	x29, sp, #0x10
1000003e8: b81fc3bf    	stur	wzr, [x29, #-0x4]
1000003ec: 52800640    	mov	w0, #0x32               ; =50
1000003f0: 97ffffdc    	bl	0x100000360 <__Z15test_goto_errori>
1000003f4: a9417bfd    	ldp	x29, x30, [sp, #0x10]
1000003f8: 910083ff    	add	sp, sp, #0x20
1000003fc: d65f03c0    	ret
