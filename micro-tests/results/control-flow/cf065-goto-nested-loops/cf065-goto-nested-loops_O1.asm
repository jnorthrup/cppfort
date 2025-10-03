
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf065-goto-nested-loops/cf065-goto-nested-loops_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_goto_nestedv>:
100000360: 52800008    	mov	w8, #0x0                ; =0
100000364: 12800000    	mov	w0, #-0x1               ; =-1
100000368: 5280000a    	mov	w10, #0x0               ; =0
10000036c: aa0a03e9    	mov	x9, x10
100000370: 7100255f    	cmp	w10, #0x9
100000374: 540000e0    	b.eq	0x100000390 <__Z16test_goto_nestedv+0x30>
100000378: 1100052a    	add	w10, w9, #0x1
10000037c: 7100151f    	cmp	w8, #0x5
100000380: 54ffff61    	b.ne	0x10000036c <__Z16test_goto_nestedv+0xc>
100000384: 7100093f    	cmp	w9, #0x2
100000388: 54ffff21    	b.ne	0x10000036c <__Z16test_goto_nestedv+0xc>
10000038c: 528006a0    	mov	w0, #0x35               ; =53
100000390: 7100253f    	cmp	w9, #0x9
100000394: 54000083    	b.lo	0x1000003a4 <__Z16test_goto_nestedv+0x44>
100000398: 11000508    	add	w8, w8, #0x1
10000039c: 7100291f    	cmp	w8, #0xa
1000003a0: 54fffe41    	b.ne	0x100000368 <__Z16test_goto_nestedv+0x8>
1000003a4: d65f03c0    	ret

00000001000003a8 <_main>:
1000003a8: 52800008    	mov	w8, #0x0                ; =0
1000003ac: 12800000    	mov	w0, #-0x1               ; =-1
1000003b0: 5280000a    	mov	w10, #0x0               ; =0
1000003b4: aa0a03e9    	mov	x9, x10
1000003b8: 7100255f    	cmp	w10, #0x9
1000003bc: 540000e0    	b.eq	0x1000003d8 <_main+0x30>
1000003c0: 1100052a    	add	w10, w9, #0x1
1000003c4: 7100151f    	cmp	w8, #0x5
1000003c8: 54ffff61    	b.ne	0x1000003b4 <_main+0xc>
1000003cc: 7100093f    	cmp	w9, #0x2
1000003d0: 54ffff21    	b.ne	0x1000003b4 <_main+0xc>
1000003d4: 528006a0    	mov	w0, #0x35               ; =53
1000003d8: 7100253f    	cmp	w9, #0x9
1000003dc: 54000083    	b.lo	0x1000003ec <_main+0x44>
1000003e0: 11000508    	add	w8, w8, #0x1
1000003e4: 7100291f    	cmp	w8, #0xa
1000003e8: 54fffe41    	b.ne	0x1000003b0 <_main+0x8>
1000003ec: d65f03c0    	ret
