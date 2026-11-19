(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* FlambdaStatedSymbolsSpec.v under a renaming: the lemmas from which
   the laws of renaming of the rewrites that read what is known of the
   symbols are proved. There is no statement here, and nothing here
   needs review.

   A renaming does not rename symbols, and what is known of a symbol
   has no names in it: what is known of the symbols
   (FlambdaStatedSymbolsSpec.symbols_stated) is the same list on both
   sides of each law. What a group of static bindings adds to it reads
   the fields that are symbols or tagged immediates, which a renaming
   leaves as they are, and says nothing of a field that is a variable,
   renamed or not. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaStatedScopeSpec FlambdaStatedSymbolsSpec
  FlambdaStatedScopeNamesProofs.
Import ListNotations.

Lemma is_stated_knowing_equivariant (r : renaming) (Y : symbols_stated)
    (S : stated_in_scope) (ps : param_subkind) (s : simple) :
  is_stated_knowing Y (rename_stated_in_scope r S) ps (rename_simple r s)
  = is_stated_knowing Y S ps s.
Proof.
  destruct s as [x | sym | c]; cbn [rename_simple is_stated_knowing];
    [ exact (is_stated_equivariant r S ps (Var x)) | reflexivity
    | exact (is_stated_equivariant r S ps (Const c)) ].
Qed.

Lemma are_stated_knowing_equivariant (r : renaming) (Y : symbols_stated)
    (S : stated_in_scope) (sks : list param_subkind) :
  ∀ args : list simple,
    are_stated_knowing Y (rename_stated_in_scope r S)
      (map (rename_simple r) args) sks
    = are_stated_knowing Y S args sks.
Proof.
  induction sks as [ | ps sks IH]; intros [ | s args]; try reflexivity.
  cbn [map are_stated_knowing].
  rewrite is_stated_knowing_equivariant, IH. reflexivity.
Qed.

Lemma static_field_stated_renamed (r : renaming) (Y : symbols_stated)
    (s : simple) :
  static_field_stated Y (rename_simple r s) = static_field_stated Y s.
Proof. destruct s as [x | sym | c]; reflexivity. Qed.

Lemma static_fields_stated_renamed (r : renaming) (Y : symbols_stated) :
  ∀ fields : list simple,
    static_fields_stated Y (map (rename_simple r) fields)
    = static_fields_stated Y fields.
Proof.
  induction fields as [ | s fields IH]; cbn [map static_fields_stated];
    [ reflexivity | ].
  rewrite static_field_stated_renamed, IH. reflexivity.
Qed.

Lemma stated_of_static_const_renamed (r : renaming) (Y : symbols_stated)
    (const : static_const) :
  stated_of_static_const Y (rename_static_const r const)
  = stated_of_static_const Y const.
Proof.
  destruct const; cbn [rename_static_const stated_of_static_const];
    try reflexivity.
  rewrite static_fields_stated_renamed. reflexivity.
Qed.

(* What a renamed group of bindings adds is what the group adds. *)
Lemma statics_stated_equivariant (r : renaming)
    (bindings : list static_binding) :
  ∀ Y : symbols_stated,
    statics_stated (map (rename_static_binding r) bindings) Y
    = statics_stated bindings Y.
Proof.
  induction bindings as [ | [id c | id | sym const] bindings IH];
    intros Y; simpl; [ reflexivity | apply IH | apply IH | ].
  rewrite stated_of_static_const_renamed. apply IH.
Qed.
