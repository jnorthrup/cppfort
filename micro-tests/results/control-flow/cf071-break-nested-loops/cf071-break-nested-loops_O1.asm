
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf071-break-nested-loops/cf071-break-nested-loops_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z17test_break_nestedv>:
100000360: 52800008    	mov	w8, #0x0                ; =0
100000364: 52800009    	mov	w9, #0x0                ; =0
100000368: 5280000a    	mov	w10, #0x0               ; =0
10000036c: 12800000    	mov	w0, #-0x1               ; =-1
100000370: 1280026b    	mov	w11, #-0x14             ; =-20
100000374: 5280000c    	mov	w12, #0x0               ; =0
100000378: aa0b03ed    	mov	x13, x11
10000037c: 2b0901ad    	adds	w13, w13, w9
100000380: 540000e0    	b.eq	0x10000039c <__Z17test_break_nestedv+0x3c>
100000384: 1100058c    	add	w12, w12, #0x1
100000388: 7100299f    	cmp	w12, #0xa
10000038c: 54ffff81    	b.ne	0x10000037c <__Z17test_break_nestedv+0x1c>
100000390: 7100213f    	cmp	w9, #0x8
100000394: 540000c9    	b.ls	0x1000003ac <__Z17test_break_nestedv+0x4c>
100000398: 14000009    	b	0x1000003bc <__Z17test_break_nestedv+0x5c>
10000039c: 4b080180    	sub	w0, w12, w8
1000003a0: 5280002a    	mov	w10, #0x1               ; =1
1000003a4: 7100213f    	cmp	w9, #0x8
1000003a8: 540000a8    	b.hi	0x1000003bc <__Z17test_break_nestedv+0x5c>
1000003ac: 11000529    	add	w9, w9, #0x1
1000003b0: 51002908    	sub	w8, w8, #0xa
1000003b4: 5100056b    	sub	w11, w11, #0x1
1000003b8: 3607fdea    	tbz	w10, #0x0, 0x100000374 <__Z17test_break_nestedv+0x14>
1000003bc: d65f03c0    	ret

00000001000003c0 <_main>:
1000003c0: 52800008    	mov	w8, #0x0                ; =0
1000003c4: 52800009    	mov	w9, #0x0                ; =0
1000003c8: 5280000a    	mov	w10, #0x0               ; =0
1000003cc: 12800000    	mov	w0, #-0x1               ; =-1
1000003d0: 1280026b    	mov	w11, #-0x14             ; =-20
1000003d4: 5280000c    	mov	w12, #0x0               ; =0
1000003d8: aa0b03ed    	mov	x13, x11
1000003dc: 2b0901ad    	adds	w13, w13, w9
1000003e0: 540000e0    	b.eq	0x1000003fc <_main+0x3c>
1000003e4: 1100058c    	add	w12, w12, #0x1
1000003e8: 7100299f    	cmp	w12, #0xa
1000003ec: 54ffff81    	b.ne	0x1000003dc <_main+0x1c>
1000003f0: 7100213f    	cmp	w9, #0x8
1000003f4: 540000c9    	b.ls	0x10000040c <_main+0x4c>
1000003f8: 14000009    	b	0x10000041c <_main+0x5c>
1000003fc: 4b080180    	sub	w0, w12, w8
100000400: 5280002a    	mov	w10, #0x1               ; =1
100000404: 7100213f    	cmp	w9, #0x8
100000408: 540000a8    	b.hi	0x10000041c <_main+0x5c>
10000040c: 11000529    	add	w9, w9, #0x1
100000410: 51002908    	sub	w8, w8, #0xa
100000414: 5100056b    	sub	w11, w11, #0x1
100000418: 3607fdea    	tbz	w10, #0x0, 0x1000003d4 <_main+0x14>
10000041c: d65f03c0    	ret
