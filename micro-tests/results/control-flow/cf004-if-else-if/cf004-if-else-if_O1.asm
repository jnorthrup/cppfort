
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf004-if-else-if/cf004-if-else-if_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z15test_if_else_ifi>:
100000360: 12800008    	mov	w8, #-0x1               ; =-1
100000364: 7100281f    	cmp	w0, #0xa
100000368: 52800029    	mov	w9, #0x1                ; =1
10000036c: 1a89b529    	cinc	w9, w9, ge
100000370: 7100001f    	cmp	w0, #0x0
100000374: 1a8903e9    	csel	w9, wzr, w9, eq
100000378: 7201001f    	tst	w0, #0x80000000
10000037c: 1a891100    	csel	w0, w8, w9, ne
100000380: d65f03c0    	ret

0000000100000384 <_main>:
100000384: 52800020    	mov	w0, #0x1                ; =1
100000388: d65f03c0    	ret
