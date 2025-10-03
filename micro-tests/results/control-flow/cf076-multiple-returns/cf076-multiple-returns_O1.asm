
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf076-multiple-returns/cf076-multiple-returns_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z21test_multiple_returnsi>:
100000360: 12800008    	mov	w8, #-0x1               ; =-1
100000364: 52800029    	mov	w9, #0x1                ; =1
100000368: 7101901f    	cmp	w0, #0x64
10000036c: 5280004a    	mov	w10, #0x2               ; =2
100000370: 1a8ab54a    	cinc	w10, w10, ge
100000374: 7100281f    	cmp	w0, #0xa
100000378: 1a8ab129    	csel	w9, w9, w10, lt
10000037c: 7100001f    	cmp	w0, #0x0
100000380: 1a8903e9    	csel	w9, wzr, w9, eq
100000384: 7201001f    	tst	w0, #0x80000000
100000388: 1a891100    	csel	w0, w8, w9, ne
10000038c: d65f03c0    	ret

0000000100000390 <_main>:
100000390: 52800040    	mov	w0, #0x2                ; =2
100000394: d65f03c0    	ret
