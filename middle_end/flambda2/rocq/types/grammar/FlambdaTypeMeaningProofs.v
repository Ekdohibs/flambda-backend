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


(* Proofs of the statements of FlambdaTypeMeaningSpec.v, and examples: for
   each constructor of the grammar but [Mutable_block], a value that has a
   type built with it and a value that has not. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaValueProofs FlambdaPrimitiveProofs FlambdaWellFormedSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the meaning reads of a value                                        *)
(* ======================================================================== *)

Lemma block_of_kind (statics : static_env) (v : value) (tag : Z)
    (fields : list value) :
  block_of statics v = Some (tag, fields) -> kind_of_value v = Value.
Proof. destruct v; try discriminate; intros _; reflexivity. Qed.

Lemma boxed_number_of_kind (statics : static_env) (v : value) (c : const) :
  boxed_number_of statics v = Some c -> kind_of_value v = Value.
Proof. destruct v; try discriminate; intros _; reflexivity. Qed.

Lemma string_of_kind (statics : static_env) (v : value) (bytes : list Z) :
  string_of statics v = Some bytes -> kind_of_value v = Value.
Proof. destruct v; try discriminate; intros _; reflexivity. Qed.

Lemma closure_of_kind (statics : static_env) (v : value)
    (closure : code_id * list (value_slot * value)) :
  closure_of statics v = Some closure -> kind_of_value v = Value.
Proof. destruct v; try discriminate; intros _; reflexivity. Qed.

Lemma closure_in_set_kind (statics : static_env) (v : value)
    (closure :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  closure_in_set statics v = Some closure -> kind_of_value v = Value.
Proof. destruct v; try discriminate; intros _; reflexivity. Qed.

(* ======================================================================== *)
(* Which value a value is                                                   *)
(* ======================================================================== *)

(* A definition by cases on the value where the meaning says "it is
   [Null]", "it is a tagged immediate" or "it is a naked immediate": the
   three lemmas below say what such a case analysis comes to, so that a
   proof need not look at the constructors of values. *)

Lemma null_or_not {A : Type} (v : value) (a b : A) :
  (v = Value_const Null
   ∧ match v with
     | Value_const Null => a
     | _ => b
     end
     = a)
  ∨ (v <> Value_const Null
     ∧ match v with
       | Value_const Null => a
       | _ => b
       end
       = b).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    [ destruct c | | | | | | | | ];
    first
      [ left; split; reflexivity
      | right; split; [ discriminate | reflexivity ] ].
Qed.

Lemma tagged_immediate_or_not {A : Type} (v : value) (a : Z -> A) (b : A) :
  (∃ i : Z,
    v = Value_const (Tagged_immediate i)
    ∧ match v with
      | Value_const (Tagged_immediate i) => a i
      | _ => b
      end
      = a i)
  ∨ ((∀ i : Z, v <> Value_const (Tagged_immediate i))
     ∧ match v with
       | Value_const (Tagged_immediate i) => a i
       | _ => b
       end
       = b).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    [ destruct c | | | | | | | | ];
    first
      [ left; eexists; split; reflexivity
      | right; split; [ intros ?; discriminate | reflexivity ] ].
Qed.

Lemma naked_immediate_or_not {A : Type} (v : value) (a : Z -> A) (b : A) :
  (∃ i : Z,
    v = Value_const (Naked_immediate i)
    ∧ match v with
      | Value_const (Naked_immediate i) => a i
      | _ => b
      end
      = a i)
  ∨ ((∀ i : Z, v <> Value_const (Naked_immediate i))
     ∧ match v with
       | Value_const (Naked_immediate i) => a i
       | _ => b
       end
       = b).
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    [ destruct c | | | | | | | | ];
    first
      [ left; eexists; split; reflexivity
      | right; split; [ intros ?; discriminate | reflexivity ] ].
Qed.

(* ======================================================================== *)
(* Maps of types                                                            *)
(* ======================================================================== *)

Theorem name_eqb_decides_equality : name_eqb_decides_equality_statement.
Proof. exact name_eqb_eq. Qed.

(* ======================================================================== *)
(* One equation for each constructor                                        *)
(* ======================================================================== *)

(* What [type_satisfied] and the functions defined with it come to on each
   constructor. They hold by definition; they are stated so that a proof
   can rewrite with them and never unfold the definition, whose functions
   are defined together and print badly when unfolded. The hint database
   [type_satisfied] has them all. The equations for the row-like
   structures, the entries of closures and the environment extensions are
   further down, by [lookup]: [row_like_for_blocks_find_iff],
   [row_like_for_closures_lookup_iff], [closures_entry_lookup_iff] and
   [env_extension_lookup_iff]. *)

Lemma type_satisfied_value (statics : static_env) (vars : env) (v : value)
    (ty : type_descr head_of_kind_value) :
  type_satisfied statics vars v (Type_value ty)
  = descr_satisfied vars Value (head_of_kind_value_satisfied statics vars v)
      v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_naked_immediate (statics : static_env) (vars : env)
    (v : value) (ty : type_descr head_of_kind_naked_immediate) :
  type_satisfied statics vars v (Type_naked_immediate ty)
  = descr_satisfied vars (Naked_number Kind_naked_immediate)
      (head_of_kind_naked_immediate_satisfied statics vars v) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_naked_number (statics : static_env) (vars : env)
    (v : value) (k : naked_number_kind) (ty : type_descr (list Z)) :
  type_satisfied statics vars v (Type_naked_number k ty)
  = descr_satisfied vars (Naked_number k) (naked_numbers_satisfied k v) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_rec_info (statics : static_env) (vars : env) (v : value)
    (ty : type_descr unit) :
  type_satisfied statics vars v (Type_rec_info ty)
  = descr_satisfied vars Rec_info (fun _ => kind_of_value v = Rec_info) v ty.
Proof. reflexivity. Qed.

Lemma type_satisfied_region (statics : static_env) (vars : env) (v : value)
    (ty : type_descr unit) :
  type_satisfied statics vars v (Type_region ty)
  = descr_satisfied vars Region (fun _ => kind_of_value v = Region) v ty.
Proof. reflexivity. Qed.

Lemma head_of_kind_value_satisfied_eq (statics : static_env) (vars : env)
    (v : value) (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null) :
  head_of_kind_value_satisfied statics vars v
    (Head_of_kind_value non_null is_null)
  = ((v = Value_const Null
      ∧ match is_null with
        | Not_null => False
        | Maybe_null is_null => variable_is vars is_null 1
        end)
     ∨ (v <> Value_const Null
        ∧ match non_null with
          | Unknown => kind_of_value v = Value
          | Bottom => False
          | Ok head =>
            head_of_kind_value_non_null_satisfied statics vars v head
          end
        ∧ match is_null with
          | Not_null => True
          | Maybe_null is_null => variable_is vars is_null 0
          end)).
Proof. reflexivity. Qed.

Lemma variant_satisfied (statics : static_env) (vars : env) (v : value)
    (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Variant is_int immediates get_tag blocks extensions is_unique)
  = ((∃ i : Z,
       v = Value_const (Tagged_immediate i)
       ∧ variable_is vars is_int 1
       ∧ match immediates with
         | Or_unknown.Unknown => True
         | Or_unknown.Known ty =>
           type_satisfied statics vars (Value_const (Naked_immediate i)) ty
         end
       ∧ match extensions with
         | No_extensions => True
         | Ext when_immediate _ =>
           env_extension_satisfied statics vars when_immediate
         end)
     ∨ (∃ (tag : Z) (fields : list value),
         block_of statics v = Some (tag, fields)
         ∧ variable_is vars is_int 0
         ∧ variable_is vars get_tag tag
         ∧ match blocks with
           | Or_unknown.Unknown => True
           | Or_unknown.Known blocks =>
             row_like_for_blocks_satisfied statics vars tag fields blocks
           end
         ∧ match extensions with
           | No_extensions => True
           | Ext _ when_block =>
             env_extension_satisfied statics vars when_block
           end)).
Proof. reflexivity. Qed.

Lemma mutable_block_head_satisfied (statics : static_env) (vars : env)
    (v : value) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Mutable_block alloc_mode)
  = False.
Proof. reflexivity. Qed.

Lemma boxed_number_satisfied (statics : static_env) (vars : env) (v : value)
    (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Boxed_number k contents alloc_mode)
  = (∃ c : const,
      boxed_number_of statics v = Some c
      ∧ kind_of_const c = unboxed_kind k
      ∧ type_satisfied statics vars (Value_const c) contents).
Proof. reflexivity. Qed.

Lemma closures_satisfied (statics : static_env) (vars : env) (v : value)
    (by_function_slot : row_like_for_closures) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_satisfied statics vars v
    (Closures by_function_slot alloc_mode)
  = (∃ (slot : function_slot) (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closure_in_set statics v = Some (slot, functions, value_slots)
      ∧ row_like_for_closures_satisfied statics vars v slot functions
          value_slots by_function_slot).
Proof. reflexivity. Qed.

Lemma string_satisfied (statics : static_env) (vars : env) (v : value)
    (strs : list (list Z)) :
  head_of_kind_value_non_null_satisfied statics vars v (String strs)
  = (∃ bytes : list Z, string_of statics v = Some bytes ∧ bytes ∈ strs).
Proof. reflexivity. Qed.

Create Rewrite HintDb type_satisfied.
#[global] Hint Rewrite type_satisfied_value type_satisfied_naked_immediate
  type_satisfied_naked_number type_satisfied_rec_info type_satisfied_region
  head_of_kind_value_satisfied_eq variant_satisfied
  mutable_block_head_satisfied boxed_number_satisfied closures_satisfied
  string_satisfied : type_satisfied.

(* ======================================================================== *)
(* What every kind has: unknown, bottom, aliases                            *)
(* ======================================================================== *)

Theorem unknown_satisfied : unknown_satisfied_statement.
Proof. intros statics vars v k. destruct k as [ | [] | | ]; reflexivity. Qed.

Theorem bottom_satisfied : bottom_satisfied_statement.
Proof.
  intros statics vars v k. destruct k as [ | [] | | ]; intros [].
Qed.

Theorem alias_satisfied : alias_satisfied_statement.
Proof. intros statics vars v k s. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma get_alias_descr {head : Type} (vars : env) (k : flambda_kind)
    (head_satisfied : head -> Prop) (v : value) (ty : type_descr head)
    (s : simple) :
  match ty with
  | Ok (Equals s) => Some s
  | Unknown | Bottom | Ok (No_alias _) => None
  end = Some s ->
  (descr_satisfied vars k head_satisfied v ty
   <-> eval_simple vars s = Some v ∧ kind_of_value v = k).
Proof.
  destruct ty as [ | [h | s'] | ]; try discriminate.
  intros [= ->]. reflexivity.
Qed.

Theorem get_alias_satisfied : get_alias_satisfied_statement.
Proof.
  intros statics vars v t s. destruct t; simpl; apply get_alias_descr.
Qed.

Theorem this_const_satisfied : this_const_satisfied_statement.
Proof.
  intros statics vars v c. unfold this_const.
  rewrite (alias_satisfied statics vars v).
  simpl. split.
  - intros [[= <-] _]. reflexivity.
  - intros ->. split; reflexivity.
Qed.

(* ======================================================================== *)
(* Null, immediates and blocks                                              *)
(* ======================================================================== *)

Theorem null_satisfied : null_satisfied_statement.
Proof.
  intros statics vars v. simpl. split.
  - intros [[Hnull _] | [_ [[] _]]]. exact Hnull.
  - intros ->. left. split; [ reflexivity | exact I ].
Qed.

Theorem any_non_null_value_satisfied : any_non_null_value_satisfied_statement.
Proof.
  intros statics vars v. simpl. split.
  - intros [[_ []] | [Hnot_null [Hkind _]]]. split; assumption.
  - intros [Hkind Hnot_null]. right. repeat split; assumption.
Qed.

Theorem type_for_const_satisfied : type_for_const_satisfied_statement.
Proof.
  intros statics vars c.
  destruct c; try (apply this_const_satisfied; reflexivity).
  - apply null_satisfied. reflexivity.
  - apply unknown_satisfied. reflexivity.
Qed.

(* A non-null head alone: the value is not [Null] for free, as no head
   admits [Null]. *)
Lemma head_non_null_kind (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  head_of_kind_value_non_null_satisfied statics vars v head ->
  kind_of_value v = Value ∧ v <> Value_const Null.
Proof.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs]; simpl.
  - intros [[i [-> _]] | [tag [fields [Hblock _]]]];
      [ split; [ reflexivity | discriminate ] | ].
    split; [ exact (block_of_kind _ _ _ _ Hblock) | ].
    intros ->. discriminate.
  - intros [].
  - intros [c [Hboxed _]].
    split; [ exact (boxed_number_of_kind _ _ _ Hboxed) | ].
    intros ->. discriminate.
  - intros [slot [functions [value_slots [Hclosure _]]]].
    split; [ exact (closure_in_set_kind _ _ _ Hclosure) | ].
    intros ->. discriminate.
  - intros [bytes [Hstring _]].
    split; [ exact (string_of_kind _ _ _ Hstring) | ].
    intros ->. discriminate.
Qed.

Lemma non_null_value_satisfied (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  type_satisfied statics vars v (non_null_value head)
  <-> head_of_kind_value_non_null_satisfied statics vars v head.
Proof.
  simpl. split.
  - intros [[_ []] | [_ [Hhead _]]]. exact Hhead.
  - intros Hhead. right.
    split; [ exact (proj2 (head_non_null_kind _ _ _ _ Hhead)) | ].
    split; [ exact Hhead | exact I ].
Qed.

Theorem type_satisfied_kind : type_satisfied_kind_statement.
Proof.
  intros statics vars v t.
  assert (Hdescr :
    ∀ (head : Type) (k : flambda_kind) (head_satisfied : head -> Prop)
      (ty : type_descr head),
      (∀ h : head, head_satisfied h -> kind_of_value v = k) ->
      descr_satisfied vars k head_satisfied v ty -> kind_of_value v = k).
  { intros head k head_satisfied [ | [h | s] | ] Hhead; simpl;
      [ auto | apply Hhead | intros [_ Hkind]; exact Hkind | intros [] ]. }
  destruct t as [ty | ty | k ty | ty | ty]; simpl; apply Hdescr.
  - intros [non_null is_null]. simpl.
    intros [[-> _] | [_ [Hnon_null _]]]; [ reflexivity | ].
    destruct non_null as [ | head | ];
      [ exact Hnon_null | | destruct Hnon_null ].
    exact (proj1 (head_non_null_kind _ _ _ _ Hnon_null)).
  - intros head [i [-> _]]. reflexivity.
  - intros numbers [c [n [-> [Hkind _]]]]. exact Hkind.
  - auto.
  - auto.
Qed.

Theorem create_variant_satisfied : create_variant_satisfied_statement.
Proof.
  intros statics vars v is_unique immediates blocks extensions.
  unfold create_variant. rewrite non_null_value_satisfied. simpl.
  split.
  - intros [[i [Hv [_ Himmediate]]] | [tag [fields [Hblock [_ [_ Hblock']]]]]].
    + left. exists i. split; assumption.
    + right. exists tag, fields. split; assumption.
  - intros [[i [Hv Himmediate]] | [tag [fields [Hblock Hblock']]]].
    + left. exists i. repeat split; tauto.
    + right. exists tag, fields. repeat split; tauto.
Qed.

Lemma row_like_for_blocks_bottom_satisfied (statics : static_env) (vars : env)
    (tag : Z) (fields : list value) :
  ~ row_like_for_blocks_satisfied statics vars tag fields
      row_like_for_blocks_bottom.
Proof. simpl. intros [_ Hother]. exact (Hother eq_refl). Qed.

Theorem any_tagged_immediate_satisfied
  : any_tagged_immediate_satisfied_statement.
Proof.
  intros statics vars v. unfold any_tagged_immediate.
  rewrite (create_variant_satisfied statics vars v). split.
  - intros [[i [Hv _]] | [tag [fields [_ [Hblocks _]]]]].
    + exists i. exact Hv.
    + destruct (row_like_for_blocks_bottom_satisfied _ _ _ _ Hblocks).
  - intros [i Hv]. left. exists i. repeat split. exact Hv.
Qed.

Theorem tag_immediate_satisfied : tag_immediate_satisfied_statement.
Proof.
  intros statics vars v t. unfold tag_immediate.
  rewrite (create_variant_satisfied statics vars v). split.
  - intros [[i [Hv [Ht _]]] | [tag [fields [_ [Hblocks _]]]]].
    + exists i. split; assumption.
    + destruct (row_like_for_blocks_bottom_satisfied _ _ _ _ Hblocks).
  - intros [i [Hv Ht]]. left. exists i. repeat split; assumption.
Qed.

Theorem any_block_satisfied : any_block_satisfied_statement.
Proof.
  intros statics vars v. unfold any_block.
  rewrite (create_variant_satisfied statics vars v). split.
  - intros [[i [_ [Hbottom _]]] | [tag [fields [Hblock _]]]].
    + destruct (bottom_satisfied _ _ _ _ Hbottom).
    + exists tag, fields. exact Hblock.
  - intros [tag [fields Hblock]]. right. exists tag, fields.
    repeat split. exact Hblock.
Qed.

Theorem mutable_block_satisfied : mutable_block_satisfied_statement.
Proof.
  intros statics vars v alloc_mode. unfold mutable_block.
  rewrite non_null_value_satisfied. simpl. auto.
Qed.

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

Theorem these_naked_immediates_satisfied
  : these_naked_immediates_satisfied_statement.
Proof.
  intros statics vars v imms.
  assert (Hhead :
    type_satisfied statics vars v
      (Type_naked_immediate
         (Ok
            (No_alias
               (Head_of_kind_naked_immediate (Or_unknown.Known imms) []))))
    <-> ∃ i : Z, v = Value_const (Naked_immediate i) ∧ i ∈ imms).
  { simpl. unfold head_of_kind_naked_immediate_satisfied. simpl. split.
    - intros [i [Hv [Hin _]]]. exists i. split; assumption.
    - intros [i [Hv Hin]]. exists i. repeat split; [ assumption .. | ].
      intros relation n []. }
  destruct imms as [ | i [ | i' imms]]; [ | | exact Hhead ].
  - split; [ intros Hbottom | intros [i [_ []]] ].
    destruct (bottom_satisfied _ _ _ _ Hbottom).
  - unfold these_naked_immediates, this_naked_immediate.
    rewrite (this_const_satisfied statics vars v). split.
    + intros ->. exists i. split; [ reflexivity | left; reflexivity ].
    + intros [i' [-> [<- | []]]]. reflexivity.
Qed.

Theorem type_naked_number_satisfied : type_naked_number_satisfied_statement.
Proof. intros statics vars v k numbers. reflexivity. Qed.

(* The type of the result of a relation on the value of a name. *)
Lemma relation_type_satisfied (statics : static_env) (vars : env) (v : value)
    (relation : Relation.t) (scrutinee : name)
    (numbers : Or_unknown.t (list Z)) :
  type_satisfied statics vars v
    (Type_naked_immediate
       (Ok
          (No_alias
             (Head_of_kind_naked_immediate numbers [(relation, scrutinee)]))))
  <-> ∃ (i : Z) (w : value),
        v = Value_const (Naked_immediate i)
        ∧ match numbers with
          | Or_unknown.Unknown => True
          | Or_unknown.Known numbers => i ∈ numbers
          end
        ∧ value_of_name vars scrutinee = Some w
        ∧ relation_holds statics relation w i.
Proof.
  simpl. unfold head_of_kind_naked_immediate_satisfied. simpl. split.
  - intros [i [Hv [Hnumbers Hrelations]]].
    destruct (Hrelations relation scrutinee (or_introl eq_refl))
      as [w [Hw Hholds]].
    exists i, w. repeat split; assumption.
  - intros [i [w [Hv [Hnumbers [Hw Hholds]]]]]. exists i.
    split; [ exact Hv | ]. split; [ exact Hnumbers | ].
    intros relation' n [[= <- <-] | []]. exists w. split; assumption.
Qed.

Theorem is_int_for_scrutinee_satisfied
  : is_int_for_scrutinee_satisfied_statement.
Proof.
  intros statics vars v scrutinee. unfold is_int_for_scrutinee.
  rewrite (relation_type_satisfied statics vars v Relation.Is_int). simpl.
  split.
  - intros [i [w [Hv [Hnumbers [Hw Hholds]]]]]. exists i, w.
    split; [ exact Hv | ]. split; [ exact Hw | exact Hholds ].
  - intros [i [w [Hv [Hw Hholds]]]]. exists i, w.
    split; [ exact Hv | ]. split; [ | split; [ exact Hw | exact Hholds ] ].
    destruct Hholds as [[_ ->] | [_ ->]]; auto.
Qed.

Theorem get_tag_for_block_satisfied : get_tag_for_block_satisfied_statement.
Proof.
  intros statics vars v block. unfold get_tag_for_block.
  rewrite (relation_type_satisfied statics vars v Relation.Get_tag). simpl.
  split.
  - intros [i [w [Hv [_ [Hw Hholds]]]]]. exists i, w. auto.
  - intros [i [w [Hv [Hw Hholds]]]]. exists i, w. auto.
Qed.

Theorem is_null_for_scrutinee_satisfied
  : is_null_for_scrutinee_satisfied_statement.
Proof.
  intros statics vars v scrutinee. unfold is_null_for_scrutinee.
  rewrite (relation_type_satisfied statics vars v Relation.Is_null). simpl.
  split.
  - intros [i [w [Hv [_ [Hw Hholds]]]]]. exists i, w. auto.
  - intros [i [w [Hv [Hw Hholds]]]]. exists i, w.
    split; [ exact Hv | ]. split; [ | split; [ exact Hw | exact Hholds ] ].
    destruct Hholds as [[_ ->] | [_ [_ ->]]]; auto.
Qed.

(* The two sides of a relation agree. *)

Theorem is_null_coherent : is_null_coherent_statement.
Proof.
  intros statics vars v non_null is_null.
  rewrite head_of_kind_value_satisfied_eq. simpl.
  intros [[-> His_null] | [Hnot_null [Hnon_null His_null]]].
  - exists 1%Z. split; [ exact His_null | ]. left. split; reflexivity.
  - exists 0%Z. split; [ exact His_null | ]. right.
    split; [ | split; [ exact Hnot_null | reflexivity ] ].
    destruct non_null as [ | head | ];
      [ exact Hnon_null | | destruct Hnon_null ].
    exact (proj1 (head_non_null_kind _ _ _ _ Hnon_null)).
Qed.

Theorem is_int_coherent : is_int_coherent_statement.
Proof.
  intros statics vars v is_int immediates get_tag blocks extensions
    is_unique.
  rewrite variant_satisfied. simpl.
  intros [[i [Hv [His_int _]]] | [tag [fields [Hblock [His_int _]]]]].
  - exists 1%Z. split; [ exact His_int | ]. left.
    split; [ exists i; exact Hv | reflexivity ].
  - exists 0%Z. split; [ exact His_int | ]. right.
    split; [ exists tag, fields; exact Hblock | reflexivity ].
Qed.

Theorem get_tag_coherent : get_tag_coherent_statement.
Proof.
  intros statics vars v is_int immediates get_tag blocks extensions
    is_unique n.
  rewrite variant_satisfied. simpl.
  intros [[i [Hv _]] | [tag [fields [Hblock [_ [Hget_tag _]]]]]] Hn Htag.
  - left. exists i. exact Hv.
  - right. rewrite Hget_tag in Hn. injection Hn as <-.
    split; [ exists fields; exact Hblock | exact Htag ].
Qed.

(* The compiler's reading of a relation on a constant. *)

Theorem relation_of_const_sound : relation_of_const_sound_statement.
Proof.
  intros statics relation c i n Hholds Hof_const.
  destruct relation; simpl in Hholds.
  - destruct Hholds as [[[= ->] ->] | [_ [Hnot_null ->]]].
    + injection Hof_const as <-. reflexivity.
    + destruct c; try (injection Hof_const as <-; reflexivity).
      destruct (Hnot_null eq_refl).
  - destruct Hholds as [[[j [= ->]] ->] | [[tag [fields Hblock]] _]].
    + injection Hof_const as <-. reflexivity.
    + discriminate Hblock.
  - discriminate Hof_const.
Qed.

Theorem is_null_of_const : is_null_of_const_statement.
Proof.
  intros statics c n Hkind. simpl. split.
  - intros [[[= ->] ->] | [_ [Hnot_null ->]]]; [ reflexivity | ].
    destruct c; try reflexivity. destruct (Hnot_null eq_refl).
  - intros Hof_const.
    destruct c; try discriminate Hkind; injection Hof_const as <-.
    + right. split; [ reflexivity | ]. split; [ discriminate | reflexivity ].
    + left. split; reflexivity.
    + right. split; [ exact Hkind | ].
      split; [ discriminate | reflexivity ].
Qed.

(* The result of a primitive is related to its argument. *)

Lemma tagged_immediate_is_immediate (v : value) :
  FlambdaPrimitiveSpec.is_tagged_immediate v = true -> is_immediate v.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    try discriminate.
  destruct c; try discriminate. intros _. eexists. reflexivity.
Qed.

Lemma allocated_not_null (statics : static_env) (v : value) :
  is_allocated statics v = true ->
  kind_of_value v = Value ∧ v <> Value_const Null.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    try discriminate; intros _; (split; [ reflexivity | discriminate ]).
Qed.

Theorem primitive_related : primitive_related_statement.
Proof.
  intros statics relation v n.
  destruct relation; simpl; rewrite eval_prim_unary; simpl.
  - unfold eval_is_null.
    destruct (null_or_not v (Some true)
                (if FlambdaPrimitiveSpec.is_tagged_immediate v
                    || is_allocated statics v
                 then Some false
                 else None))
      as [[-> _] | [Hnot_null ->]].
    + simpl. intros [= <-] _. left. split; reflexivity.
    + destruct (FlambdaPrimitiveSpec.is_tagged_immediate v) eqn:Himmediate;
        simpl.
      * intros [= <-] _. right.
        destruct (tagged_immediate_is_immediate v Himmediate) as [i ->].
        split; [ reflexivity | ]. split; [ discriminate | reflexivity ].
      * destruct (is_allocated statics v) eqn:Hallocated; [ | discriminate ].
        intros [= <-] _. right.
        destruct (allocated_not_null statics v Hallocated) as [Hkind _].
        split; [ exact Hkind | ]. split; [ exact Hnot_null | reflexivity ].
  - unfold eval_is_int.
    destruct (FlambdaPrimitiveSpec.is_tagged_immediate v) eqn:Himmediate;
      simpl.
    + intros [= <-] _. left.
      split; [ exact (tagged_immediate_is_immediate v Himmediate) | ].
      reflexivity.
    + destruct (block_of statics v) as [[tag fields] | ] eqn:Hblock;
        [ | discriminate ].
      simpl. intros [= <-] _. right.
      split; [ exists tag, fields; exact Hblock | reflexivity ].
  - destruct (block_of statics v) as [[tag fields] | ]; [ | discriminate ].
    simpl. intros [= <-] Htag. right.
    split; [ exists fields; reflexivity | exact (Htag eq_refl) ].
Qed.

Theorem these_tagged_immediates_satisfied
  : these_tagged_immediates_satisfied_statement.
Proof.
  intros statics vars v imms.
  destruct imms as [ | i [ | i' imms]].
  - split; [ intros Hbottom | intros [i [_ []]] ].
    destruct (bottom_satisfied _ _ _ _ Hbottom).
  - unfold these_tagged_immediates, this_tagged_immediate.
    rewrite (this_const_satisfied statics vars v). split.
    + intros ->. exists i. split; [ reflexivity | left; reflexivity ].
    + intros [i' [-> [<- | []]]]. reflexivity.
  - unfold these_tagged_immediates.
    rewrite (tag_immediate_satisfied statics vars v). split.
    + intros [n [Hv Hn]]. exists n. split; [ exact Hv | ].
      apply (these_naked_immediates_satisfied statics vars) in Hn
        as [n' [[= <-] Hin]].
      exact Hin.
    + intros [n [Hv Hin]]. exists n. split; [ exact Hv | ].
      apply (these_naked_immediates_satisfied statics vars). exists n.
      split; [ reflexivity | exact Hin ].
Qed.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Lemma fields_satisfied_nth_iff (P : value -> flambda_type -> Prop)
    (tys : list flambda_type) (fields : list value) :
  fields_satisfied P tys fields
  <-> (∀ (i : nat) (ty : flambda_type) (field : value),
        nth_error tys i = Some ty ->
        nth_error fields i = Some field ->
        P field ty).
Proof.
  revert fields. induction tys as [ | ty tys IH]; intros fields.
  - simpl. split; [ | auto ]. intros _ [ | i] ty field; discriminate.
  - destruct fields as [ | field fields]; simpl.
    + split; [ | auto ]. intros _ [ | i] ty' field; discriminate.
    + rewrite IH. split.
      * intros [Hfield Hfields] [ | i] ty' field'; simpl.
        -- intros [= <-] [= <-]. exact Hfield.
        -- apply Hfields.
      * intros Hall. split; [ exact (Hall 0 ty field eq_refl eq_refl) | ].
        intros i ty' field'. exact (Hall (S i) ty' field').
Qed.

Theorem fields_satisfied_nth : fields_satisfied_nth_statement.
Proof. exact fields_satisfied_nth_iff. Qed.

(* With as many fields as types, position by position is [Forall2]. *)
Lemma fields_satisfied_Forall2 (P : value -> flambda_type -> Prop)
    (tys : list flambda_type) (fields : list value) :
  length fields = length tys ->
  (fields_satisfied P tys fields <-> Forall2 P fields tys).
Proof.
  revert fields. induction tys as [ | ty tys IH]; intros [ | field fields];
    simpl; try discriminate.
  - intros _. split; [ constructor | auto ].
  - intros [= Hlength]. rewrite (IH fields Hlength). split.
    + intros [Hfield Hfields]. constructor; assumption.
    + intros Hall. inversion Hall; subst. split; assumption.
Qed.

(* The case of a tag, or failing one something else. *)
Lemma at_tag_or_else {A : Type} (P : A -> Prop) (Q : Prop) (tag : Z)
    (cases : Keyed.finite_map Z A) :
  Keyed.at_binding Z.eqb P tag cases
  ∧ (Keyed.lookup Z.eqb tag cases = None -> Q)
  <-> match Keyed.lookup Z.eqb tag cases with
      | Some a => P a
      | None => Q
      end.
Proof.
  rewrite (keyed_at_binding_lookup Z Z.eqb).
  destruct (Keyed.lookup Z.eqb tag cases) as [a | ]; split.
  - intros [Hcase _]. exact (Hcase a eq_refl).
  - intros Hcase. split; [ | discriminate ]. intros a' [= <-]. exact Hcase.
  - intros [_ Hother]. exact (Hother eq_refl).
  - intros Hother. split; [ discriminate | ]. intros _. exact Hother.
Qed.

Lemma row_like_for_blocks_find_iff (statics : static_env) (vars : env)
    (tag : Z) (fields : list value)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_satisfied statics vars tag fields
    (Row_like_for_blocks known_tags other_tags alloc_mode)
  <-> match Keyed.lookup Z.eqb tag known_tags with
      | Some (Or_unknown.Known case) =>
        row_like_block_case_satisfied statics vars tag fields case
      | Some Or_unknown.Unknown => True
      | None =>
        match other_tags with
        | Or_bottom.Ok case =>
          row_like_block_case_satisfied statics vars tag fields case
        | Or_bottom.Bottom => False
        end
      end.
Proof.
  simpl. rewrite at_tag_or_else. unfold row_like_block_case in *.
  destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ];
    [ reflexivity | reflexivity | destruct other_tags; reflexivity ].
Qed.

Theorem row_like_for_blocks_find : row_like_for_blocks_find_statement.
Proof. exact row_like_for_blocks_find_iff. Qed.

Theorem immutable_block_satisfied : immutable_block_satisfied_statement.
Proof.
  intros statics vars v is_unique tag shape alloc_mode field_tys.
  unfold immutable_block.
  rewrite (create_variant_satisfied statics vars v).
  unfold row_like_for_blocks_create_exactly.
  split.
  - intros [[i [_ [Hbottom _]]] | [tag' [fields [Hblock [Hblocks _]]]]];
      [ destruct (bottom_satisfied _ _ _ _ Hbottom) | ].
    rewrite row_like_for_blocks_find_iff in Hblocks. simpl in Hblocks.
    destruct (Z.eqb_spec tag' tag) as [-> | _]; [ | destruct Hblocks ].
    destruct Hblocks as [[Hlength Hshape] [Hfields _]]. simpl in *.
    exists fields. split; [ exact Hblock | ]. split; [ exact Hshape | ].
    apply fields_satisfied_Forall2; assumption.
  - intros [fields [Hblock [Hshape Hfields]]]. right. exists tag, fields.
    split; [ exact Hblock | ]. split; [ | exact I ].
    rewrite row_like_for_blocks_find_iff. simpl. rewrite Z.eqb_refl.
    assert (Hlength : length fields = length field_tys)
      by exact (Forall2_length Hfields).
    split; [ split; [ exact Hlength | exact Hshape ] | ].
    split; [ apply fields_satisfied_Forall2; assumption | ].
    simpl. split; exact I.
Qed.

(* ======================================================================== *)
(* Boxed numbers, strings and closures                                      *)
(* ======================================================================== *)

Theorem box_number_satisfied : box_number_satisfied_statement.
Proof.
  intros statics vars v k contents alloc_mode. unfold box_number.
  rewrite non_null_value_satisfied. reflexivity.
Qed.

Theorem this_immutable_string_satisfied
  : this_immutable_string_satisfied_statement.
Proof.
  intros statics vars v bytes. unfold this_immutable_string.
  rewrite non_null_value_satisfied. simpl. split.
  - intros [bytes' [Hstring [<- | []]]]. exact Hstring.
  - intros Hstring. exists bytes. split; [ exact Hstring | ].
    left. reflexivity.
Qed.

Lemma row_like_for_closures_lookup_iff (statics : static_env) (vars : env)
    (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (known_closures : finite_map row_like_closures_case) :
  row_like_for_closures_satisfied statics vars v slot functions value_slots
    (Row_like_for_closures known_closures)
  <-> ∃ case : row_like_closures_case,
        lookup slot known_closures = Some case
        ∧ row_like_closures_case_satisfied statics vars v slot functions
            value_slots case.
Proof.
  simpl. rewrite (at_binding_lookup row_like_closures_case), is_bound_lookup.
  split.
  - intros [[case Hlookup] Hcase]. exists case.
    split; [ exact Hlookup | exact (Hcase case Hlookup) ].
  - intros [case [Hlookup Hcase]]. split; [ exists case; exact Hlookup | ].
    intros case' Hlookup'. rewrite Hlookup in Hlookup'.
    injection Hlookup' as <-. exact Hcase.
Qed.

Theorem row_like_for_closures_lookup
  : row_like_for_closures_lookup_statement.
Proof. exact row_like_for_closures_lookup_iff. Qed.

Lemma closures_entry_lookup_iff (statics : static_env) (vars : env)
    (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  closures_entry_satisfied statics vars v slot functions value_slots
    (Closures_entry function_types closure_types value_slot_types)
  <-> (∀ (g : function_slot) (function_type : function_type)
         (code : code_id),
        lookup g function_types = Some (Or_unknown.Known function_type) ->
        lookup g functions = Some code ->
        code = function_type_code_id function_type)
      ∧ (∀ (g : function_slot) (ty : flambda_type) (closure : value),
          lookup g closure_types = Some ty ->
          project_function_slot statics slot g v = Some closure ->
          type_satisfied statics vars closure ty)
      ∧ (∀ (w : value_slot) (ty : flambda_type) (contents : value),
          lookup w value_slot_types = Some ty ->
          lookup w value_slots = Some contents ->
          type_satisfied statics vars contents ty).
Proof.
  simpl. rewrite !(for_all_bindings_lookup flambda_type).
  split; intros [Hfunctions [Hclosures Hslots]];
    (split; [ exact Hfunctions | split; eauto ]).
Qed.

Theorem closures_entry_lookup : closures_entry_lookup_statement.
Proof. exact closures_entry_lookup_iff. Qed.

Theorem closures_case_function_code : closures_case_function_code_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension g function_type
    [Hindex [Hentry _]] Hamong Hfunction_type.
  simpl in Hindex, Hentry.
  apply closures_entry_lookup_iff in Hentry as [Hfunctions _].
  assert (Hbound : is_bound g functions = true).
  { rewrite <- among_domain. unfold closures_index_satisfied in Hindex.
    unfold closures_index_contents in Hamong.
    destruct (index_domain index) as [contents | contents].
    - destruct Hindex as [Hsame _]. rewrite <- (Hsame g). exact Hamong.
    - destruct Hindex as [Hall _].
      exact (proj1 (all_among _ _) Hall g Hamong). }
  unfold is_bound in Hbound.
  destruct (lookup g functions) as [code | ] eqn:Hcode;
    [ | discriminate Hbound ].
  rewrite (Hfunctions g function_type code Hfunction_type Hcode).
  reflexivity.
Qed.

Theorem closures_case_own_code : closures_case_own_code_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension function_type Hclosure
    [_ [Hentry _]] Hfunction_type.
  simpl in Hentry.
  apply closures_entry_lookup_iff in Hentry as [Hfunctions _].
  pose proof (closure_in_set_slot _ _ _ _ _ Hclosure) as Hbound.
  unfold is_bound in Hbound.
  destruct (lookup slot functions) as [code | ] eqn:Hcode;
    [ | discriminate Hbound ].
  rewrite (Hfunctions slot function_type code Hfunction_type Hcode).
  reflexivity.
Qed.

Lemma env_extension_lookup_iff (statics : static_env) (vars : env)
    (equations : Keyed.finite_map name flambda_type) :
  env_extension_satisfied statics vars (Env_extension equations)
  <-> (∀ (n : name) (ty : flambda_type),
        Keyed.lookup name_eqb n equations = Some ty ->
        ∃ v : value,
          value_of_name vars n = Some v ∧ type_satisfied statics vars v ty).
Proof.
  simpl. rewrite (keyed_for_all_bindings_lookup name name_eqb name_eqb_eq).
  reflexivity.
Qed.

Theorem env_extension_lookup : env_extension_lookup_statement.
Proof. exact env_extension_lookup_iff. Qed.

Lemma empty_env_extension_satisfied (statics : static_env) (vars : env) :
  env_extension_satisfied statics vars empty_env_extension.
Proof.
  apply env_extension_lookup_iff. intros [x | s] ty; discriminate.
Qed.

Theorem exactly_this_closure_satisfied
  : exactly_this_closure_satisfied_statement.
Proof.
  intros statics vars v slot function_types closure_types value_slot_types
    alloc_mode.
  unfold exactly_this_closure, create_closures.
  rewrite non_null_value_satisfied, closures_satisfied. split.
  - intros [slot' [functions [value_slots [Hclosure Hrow]]]].
    rewrite row_like_for_closures_lookup_iff in Hrow.
    destruct Hrow as [case [Hlookup Hcase]]. simpl in Hlookup.
    destruct (Pos.eqb_spec slot' slot) as [-> | _]; [ | discriminate ].
    injection Hlookup as <-.
    destruct Hcase as [[Hfunctions Hslots] [Hentry _]].
    exists functions, value_slots.
    split; [ exact Hclosure | ]. split; [ exact Hfunctions | ].
    split; [ exact Hslots | exact Hentry ].
  - intros [functions [value_slots [Hclosure [Hfunctions [Hslots Hentry]]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    rewrite row_like_for_closures_lookup_iff. eexists. split.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
    + split; [ split; assumption | ].
      split; [ exact Hentry | apply empty_env_extension_satisfied ].
Qed.

Theorem at_least_this_closure_satisfied
  : at_least_this_closure_satisfied_statement.
Proof.
  intros statics vars v slot function_types closure_types value_slot_types
    alloc_mode.
  unfold at_least_this_closure, create_closures.
  rewrite non_null_value_satisfied, closures_satisfied. split.
  - intros [slot' [functions [value_slots [Hclosure Hrow]]]].
    rewrite row_like_for_closures_lookup_iff in Hrow.
    destruct Hrow as [case [Hlookup Hcase]]. simpl in Hlookup.
    destruct (Pos.eqb_spec slot' slot) as [-> | _]; [ | discriminate ].
    injection Hlookup as <-.
    destruct Hcase as [[Hfunctions Hslots] [Hentry _]].
    exists functions, value_slots.
    split; [ exact Hclosure | ]. split; [ exact Hfunctions | ].
    split; [ exact Hslots | exact Hentry ].
  - intros [functions [value_slots [Hclosure [Hfunctions [Hslots Hentry]]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    rewrite row_like_for_closures_lookup_iff. eexists. split.
    + simpl. rewrite Pos.eqb_refl. reflexivity.
    + split; [ split; assumption | ].
      split; [ exact Hentry | apply empty_env_extension_satisfied ].
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* For each constructor of the grammar but one, a value that has a type
   built with it and a value that has not, under a static environment
   with blocks of the three shapes, two boxed numbers, a string and a
   closure, and an environment with four variables. A type that is shown
   to reject a value is one that some other value has, but where it is
   said not to be. The exception is [Mutable_block]: the machine has no
   mutable block, so no value has such a type
   ([mutable_block_satisfied]), and what is stated of it holds for want
   of values until there is a heap.

   What the examples cover: [Unknown] and aliases; sets of naked
   immediates and the three relations, with the two cases found in
   review (a parameter that is unboxed and holds an immediate, and
   [Is_null] of poison and of a symbol that is not bound) and the two
   places where the compiler goes further than the meaning here (the
   block that [Meet] makes of the scrutinee of [Get_tag], and the types
   that unboxing gives to poison); two types that the n-way join made
   before f489f9445e
   (one of a function that a set of closures has not, which the closure
   has, and one that is no upper bound); two types that the
   creation functions do not make (a product wider than its index, a
   mixed record with no flat suffix); two kinds of
   the other naked numbers; null and its variable; variants with
   immediates, with blocks of a known tag, of other tags and of a tag
   with nothing known, with the variables and the extensions; the three
   shapes of blocks; two kinds of boxed numbers; strings; closures with
   exactly and with at least some contents, the code of the function,
   the type of a value slot and the type of a closure of the set, for
   a set of one function with no slot named and for a set of two
   functions, made by a [Let] and static. Not covered: the kinds of
   naked numbers and of boxed numbers other than the two shown of each,
   which the definitions treat alike. *)

Module Examples.

Local Open Scope Z_scope.

Definition a_block : symbol := 1%positive.
Definition a_boxed_float : symbol := 2%positive.
Definition a_string : symbol := 3%positive.
Definition a_closure : symbol := 4%positive.
Definition unbound : symbol := 5%positive.
Definition a_float_block : symbol := 6%positive.
Definition a_mixed_block : symbol := 7%positive.
Definition a_boxed_int64 : symbol := 8%positive.
Definition some_code : code_id := 1%positive.
Definition other_code : code_id := 2%positive.
Definition a_value_slot : value_slot := 1%positive.
Definition other_value_slot : value_slot := 2%positive.
Definition other_function_slot : function_slot := 2%positive.

(* A block of tag 0 with two fields that are values; a boxed float; a
   string of two bytes; a closure with one value slot; a block of two
   naked floats; a block of tag 0 with a value and a naked float; a
   boxed 64-bit integer. The symbol [unbound] is bound to nothing. *)
Definition statics : static_env :=
  Static_env []
    [(a_block,
      Static_block 0
        [Value_const (Tagged_immediate 7); Value_symbol a_boxed_float]);
     (a_boxed_float, Static_boxed_number (Value_const (Naked_float 42)));
     (a_string, Static_string [104; 105]);
     (a_closure,
      Static_closure some_code
        [(a_value_slot, Value_const (Tagged_immediate 3))]);
     (a_float_block,
      Static_block double_array_tag
        [Value_const (Naked_float 1); Value_const (Naked_float 2)]);
     (a_mixed_block,
      Static_block 0
        [Value_const (Tagged_immediate 7); Value_const (Naked_float 3)]);
     (a_boxed_int64, Static_boxed_number (Value_const (Naked_int64 9)))].

Definition x : variable := 1%positive.
Definition zero : variable := 2%positive.
Definition one : variable := 3%positive.
Definition poison : variable := 4%positive.

(* [x] holds a tagged immediate, [zero] and [one] the naked immediates 0
   and 1, and [poison] a poison value. *)
Definition vars : env :=
  [(x, Value_const (Tagged_immediate 5));
   (zero, Value_const (Naked_immediate 0));
   (one, Value_const (Naked_immediate 1));
   (poison, Value_const (Poison Value))].

Definition has (v : value) (t : flambda_type) : Prop :=
  type_satisfied statics vars v t.

Definition tagged (i : Z) : value := Value_const (Tagged_immediate i).
Definition naked (i : Z) : value := Value_const (Naked_immediate i).

(* What is to be proved is decided by computing: [yes] proves that a
   value has a type and [no] that it has not. *)
Ltac yes :=
  compute;
  repeat
    first
      [ exact I | reflexivity
      | match goal with
        | H : False |- _ => destruct H
        | H : _ ∨ _ |- _ => destruct H
        | H : ∃ _, _ |- _ => destruct H
        | H : ?a = ?a |- _ => clear H
        | H : _ = _ |- _ => discriminate H
        | H : (_, _) = (_, _) |- _ => injection H as <- <-
        | H : Some _ = Some _ |- _ => injection H as <-
        | H : (if ?b then _ else _) = Some _ |- _ => destruct b
        end
      | discriminate | progress compute in * | intro | split | eexists
      | left; solve [ yes ] | right; solve [ yes ] ].

Ltac no :=
  compute; intros Hsatisfied;
  repeat
    match goal with
    | H : _ ∧ _ |- _ => destruct H
    | H : ∃ _, _ |- _ => destruct H
    | H : _ ∨ _ |- _ => destruct H
    | H : False |- _ => destruct H
    | H : True |- _ => clear H
    | H : _ = _ |- _ => discriminate H
    | H : Some _ = Some _ |- _ => injection H as H; subst
    | H : Value_const _ = Value_const _ |- _ => injection H as H; subst
    | H : ?a = ?a -> _ |- _ => specialize (H eq_refl)
    | H : ?a <> ?a |- _ => destruct (H eq_refl)
    | H : (∃ k : flambda_kind, _) -> False |- _ =>
      destruct H; eexists; reflexivity
    | H : ∀ r n, (?r0, ?n0) = (r, n) ∨ _ -> _ |- _ =>
      specialize (H r0 n0 (or_introl eq_refl))
    | H : ∀ c, Some ?v = Some c -> _ |- _ => specialize (H v eq_refl)
    | H : ∀ (g : positive) (function_type : function_type) (code : positive),
        _ |- _ =>
      specialize (H 1%positive _ _ eq_refl eq_refl)
    | H : ∀ name : positive, false = _ |- _ => specialize (H 1%positive)
    | _ => progress compute in *
    end;
  try congruence; try lia.

(* [Unknown]. *)
Example unknown_has : has (tagged 1) (unknown Value).
Proof. yes. Qed.

Example unknown_has_not : ~ has (naked 1) (unknown Value).
Proof. no. Qed.

(* [Equals]. *)
Example alias_has : has (tagged 5) (alias_type_of Value (Var x)).
Proof. yes. Qed.

Example alias_has_not : ~ has (tagged 6) (alias_type_of Value (Var x)).
Proof. no. Qed.

Example alias_of_symbol_has
  : has (Value_symbol a_block) (alias_type_of Value (Symbol a_block)).
Proof. yes. Qed.

Example alias_of_symbol_has_not
  : ~ has (Value_symbol a_string) (alias_type_of Value (Symbol a_block)).
Proof. no. Qed.

(* [Type_naked_immediate]: numbers, and relations. *)
Example naked_immediates_has : has (naked 2) (these_naked_immediates [1; 2]).
Proof. yes. Qed.

Example naked_immediates_has_not
  : ~ has (naked 3) (these_naked_immediates [1; 2]).
Proof. no. Qed.

Example is_int_has : has (naked 1) (is_int_for_scrutinee (Name_var x)).
Proof. yes. Qed.

Example is_int_has_not : ~ has (naked 0) (is_int_for_scrutinee (Name_var x)).
Proof. no. Qed.

Example is_int_of_block_has
  : has (naked 0) (is_int_for_scrutinee (Name_symbol a_block)).
Proof. yes. Qed.

Example is_int_of_closure_has_not
  : ~ has (naked 0) (is_int_for_scrutinee (Name_symbol a_closure)).
Proof. no. Qed.

Example get_tag_has
  : has (naked 0) (get_tag_for_block (Name_symbol a_block)).
Proof. yes. Qed.

Example get_tag_has_not
  : ~ has (naked 1) (get_tag_for_block (Name_symbol a_block)).
Proof. no. Qed.

Example get_tag_of_closure_has_not
  : ~ has (naked 0) (get_tag_for_block (Name_symbol a_closure)).
Proof. no. Qed.

(* A parameter that is unboxed as a variant, where it holds an
   immediate: [x] is the parameter, [one] its extra parameter for
   [Is_int] and [zero] the one for [Get_tag], to which the tag 0 is
   passed on this path (unboxing_epa.ml). The types that
   build_unboxing_denv.ml records for the three all hold: the tag is
   [Get_tag] of [x], [Is_int] of [x] is 1, and [x] is a variant with
   these two variables. *)
Example unboxed_immediate_tag_has
  : has (naked 0) (get_tag_for_block (Name_var x)).
Proof. yes. Qed.

Example unboxed_immediate_is_int_has
  : has (naked 1) (is_int_for_scrutinee (Name_var x)).
Proof. yes. Qed.

Example unboxed_immediate_has
  : has (tagged 5)
      (non_null_value
         (Variant (Some one) Or_unknown.Unknown (Some zero)
            Or_unknown.Unknown No_extensions false)).
Proof. yes. Qed.

Example unboxed_immediate_is_int_has_not
  : ~ has (tagged 5)
        (non_null_value
           (Variant (Some zero) Or_unknown.Unknown (Some zero)
              Or_unknown.Unknown No_extensions false)).
Proof. no. Qed.

(* On that path the tag variable holds one of the tags [0], and [x] is
   no block of one of them: what [Meet] deduces of the scrutinee from a
   [Get_tag] relation whose number is among some tags
   ([reduce_inverse_relations]) is not true there. *)
Example unboxed_immediate_tag_among_has
  : has (naked 0) (these_naked_immediates [0]).
Proof. yes. Qed.

Example unboxed_immediate_block_has_not
  : ~ has (tagged 5) (blocks_with_these_tags [0] Alloc_mode.Heap_or_local).
Proof. no. Qed.

(* A field that is unboxed in turn as a variant, on a path where the
   parameter it is a field of holds an immediate: [poison] is the
   parameter for the field, and [zero] stands for its extra parameters
   for [Is_int] and [Get_tag], to which 0 is passed on this path. None
   of the three types that build_unboxing_denv.ml records holds there:
   the two relations hold of no value that is poison, and a poison
   value is no variant. That is for what satisfies an environment to
   set aside (see the header of the Spec, on poison). *)
Example unboxed_poison_is_int_has_not
  : ~ has (naked 0) (is_int_for_scrutinee (Name_var poison)).
Proof. no. Qed.

Example unboxed_poison_tag_has_not
  : ~ has (naked 0) (get_tag_for_block (Name_var poison)).
Proof. no. Qed.

Example unboxed_poison_has_not
  : ~ has (Value_const (Poison Value))
        (non_null_value
           (Variant (Some zero) Or_unknown.Unknown (Some zero)
              Or_unknown.Unknown No_extensions false)).
Proof. no. Qed.

Example is_null_has
  : has (naked 0) (is_null_for_scrutinee (Name_symbol a_closure)).
Proof. yes. Qed.

Example is_null_has_not
  : ~ has (naked 1) (is_null_for_scrutinee (Name_symbol a_closure)).
Proof. no. Qed.

(* [Is_null] of poison and of a symbol that is not bound is 0, as the
   variable of the type of the scrutinee says ([is_null_variable_...]
   below). *)
Example is_null_of_poison_has
  : has (naked 0) (is_null_for_scrutinee (Name_var poison)).
Proof. yes. Qed.

Example is_null_of_poison_has_not
  : ~ has (naked 1) (is_null_for_scrutinee (Name_var poison)).
Proof. no. Qed.

Example is_null_of_unbound_symbol_has
  : has (naked 0) (is_null_for_scrutinee (Name_symbol unbound)).
Proof. yes. Qed.

Example is_null_of_naked_number_has_not
  : ~ has (naked 0) (is_null_for_scrutinee (Name_var zero)).
Proof. no. Qed.

(* [Type_naked_number]. *)
Example naked_floats_has
  : has (Value_const (Naked_float 42))
      (Type_naked_number Kind_naked_float (Ok (No_alias [42; 43]))).
Proof. yes. Qed.

Example naked_floats_has_not
  : ~ has (Value_const (Naked_int32 42))
        (Type_naked_number Kind_naked_float (Ok (No_alias [42; 43]))).
Proof. no. Qed.

Example naked_int32s_has
  : has (Value_const (Naked_int32 42))
      (Type_naked_number Kind_naked_int32 (Ok (No_alias [42; 43]))).
Proof. yes. Qed.

Example naked_int32s_has_not
  : ~ has (Value_const (Naked_int32 44))
        (Type_naked_number Kind_naked_int32 (Ok (No_alias [42; 43]))).
Proof. no. Qed.

(* [Type_region] and [Type_rec_info]: the machine has no value of these
   kinds but [Poison]. *)
Example region_has
  : has (Value_const (Poison Region)) (Type_region (Ok (No_alias tt))).
Proof. yes. Qed.

Example region_has_not : ~ has (tagged 0) (Type_region (Ok (No_alias tt))).
Proof. no. Qed.

Example rec_info_has
  : has (Value_const (Poison Rec_info)) (Type_rec_info (Ok (No_alias tt))).
Proof. yes. Qed.

Example rec_info_has_not
  : ~ has (tagged 0) (Type_rec_info (Ok (No_alias tt))).
Proof. no. Qed.

(* [Head_of_kind_value]: null or not. *)
Example null_has : has (Value_const Null) null.
Proof. yes. Qed.

Example null_has_not : ~ has (tagged 0) null.
Proof. no. Qed.

Example non_null_has : has (Value_symbol unbound) any_non_null_value.
Proof. yes. Qed.

Example non_null_has_not : ~ has (Value_const Null) any_non_null_value.
Proof. no. Qed.

(* With a variable for [Is_null]: [one] holds 1 and [zero] holds 0. *)
Definition maybe_null (is_null : variable) : flambda_type :=
  Type_value
    (Ok (No_alias (Head_of_kind_value Unknown (Maybe_null (Some is_null))))).

Example is_null_variable_has : has (Value_const Null) (maybe_null one).
Proof. yes. Qed.

Example is_null_variable_has_not : ~ has (Value_const Null) (maybe_null zero).
Proof. no. Qed.

Example is_not_null_variable_has : has (tagged 0) (maybe_null zero).
Proof. yes. Qed.

Example is_not_null_variable_has_not : ~ has (tagged 0) (maybe_null one).
Proof. no. Qed.

Example is_null_variable_of_poison_has
  : has (Value_const (Poison Value)) (maybe_null zero).
Proof. yes. Qed.

Example is_null_variable_of_unbound_symbol_has
  : has (Value_symbol unbound) (maybe_null zero).
Proof. yes. Qed.

(* [Variant]: immediates. *)
Example tagged_immediates_has
  : has (tagged 1) (these_tagged_immediates [0; 1; 2]).
Proof. yes. Qed.

Example tagged_immediates_has_not
  : ~ has (tagged 5) (these_tagged_immediates [0; 1; 2]).
Proof. no. Qed.

Example any_tagged_immediate_has_not
  : ~ has (Value_symbol a_block) any_tagged_immediate.
Proof. no. Qed.

(* [Variant]: blocks, of a known tag and size. *)
Example block_has
  : has (Value_symbol a_block)
      (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
         [this_tagged_immediate 7; unknown Value]).
Proof. yes. Qed.

Example block_field_has_not
  : ~ has (Value_symbol a_block)
        (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
           [this_tagged_immediate 8; unknown Value]).
Proof. no. Qed.

Example block_size_has_not
  : ~ has (Value_symbol a_block)
        (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
           [this_tagged_immediate 7]).
Proof. no. Qed.

Example block_tag_has_not
  : ~ has (Value_symbol a_block)
        (immutable_block false 1 (Scannable Value_only) Alloc_mode.Heap
           [this_tagged_immediate 7; unknown Value]).
Proof. no. Qed.

(* The shapes. A block of values has not the type of a mixed block of
   its tag and size, which [a_mixed_block] has, and the other way
   round; a block of naked floats has the type of a float record. *)
Definition mixed_block_type : flambda_type :=
  immutable_block false 0
    (Scannable (Mixed_record 1 [Kind_naked_float])) Alloc_mode.Heap
    [unknown Value; unknown (Naked_number Kind_naked_float)].

Definition block_of_values_type : flambda_type :=
  immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
    [unknown Value; unknown Value].

Example mixed_block_has : has (Value_symbol a_mixed_block) mixed_block_type.
Proof. yes. Qed.

Example mixed_block_shape_has_not
  : ~ has (Value_symbol a_block) mixed_block_type.
Proof. no. Qed.

Example block_of_values_has : has (Value_symbol a_block) block_of_values_type.
Proof. yes. Qed.

Example block_of_values_shape_has_not
  : ~ has (Value_symbol a_mixed_block) block_of_values_type.
Proof. no. Qed.

Definition float_block_type (first : Z) : flambda_type :=
  immutable_block false double_array_tag Float_record Alloc_mode.Heap
    [Type_naked_number Kind_naked_float (Ok (No_alias [first]));
     unknown (Naked_number Kind_naked_float)].

Example float_block_has
  : has (Value_symbol a_float_block) (float_block_type 1).
Proof. yes. Qed.

Example float_block_field_has_not
  : ~ has (Value_symbol a_float_block) (float_block_type 2).
Proof. no. Qed.

Example float_block_of_values_has_not
  : ~ has (Value_symbol a_block) (float_block_type 1).
Proof. no. Qed.

Example any_block_has : has (Value_symbol a_block) any_block.
Proof. yes. Qed.

Example any_block_has_not : ~ has (Value_symbol a_closure) any_block.
Proof. no. Qed.

(* Blocks of any tag with at least so many fields: [other_tags]. *)
Definition block_of_size_at_least (size : nat) : flambda_type :=
  create_variant false
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    (Or_unknown.Known
       (row_like_for_blocks_create_at_least_unknown_tag size
          (Scannable Value_only) [this_tagged_immediate 7]
          Alloc_mode.Heap))
    No_extensions.

Example other_tags_has : has (Value_symbol a_block) (block_of_size_at_least 1).
Proof. yes. Qed.

Example other_tags_has_not
  : ~ has (Value_symbol a_block) (block_of_size_at_least 3).
Proof. no. Qed.

(* Two types that the creation functions of the compiler do not make,
   and what the meaning says of them (see the header of the Spec, on
   what the operations rely on). A product wider than the size of its
   index says nothing of a field that a block has not: a block of two
   fields has the type of the blocks of at least two fields whose third
   field has no value. And a mixed record with no flat suffix is a
   block of values. *)
Example wide_product_has
  : has (Value_symbol a_block)
      (create_variant false
         (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
         (Or_unknown.Known
            (row_like_for_blocks_create_at_least 0 2 (Scannable Value_only)
               [unknown Value; unknown Value; bottom Value]
               Alloc_mode.Heap))
         No_extensions).
Proof. yes. Qed.

Example mixed_record_of_values_has
  : has (Value_symbol a_block)
      (immutable_block false 0 (Scannable (Mixed_record 2 []))
         Alloc_mode.Heap [unknown Value; unknown Value]).
Proof. yes. Qed.

(* Blocks of a tag of which nothing more is known. *)
Definition block_of_tag (tag : Z) : flambda_type :=
  create_variant false
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
    (Or_unknown.Known
       (Row_like_for_blocks [(tag, Or_unknown.Unknown)] Or_bottom.Bottom
          Alloc_mode.Heap))
    No_extensions.

Example unknown_case_has : has (Value_symbol a_block) (block_of_tag 0).
Proof. yes. Qed.

Example unknown_case_has_not : ~ has (Value_symbol a_block) (block_of_tag 1).
Proof. no. Qed.

(* With variables for [Is_int] and [Get_tag], and with extensions: a
   block has the type only if the variables hold 0 and its tag, and if
   the equation that holds of blocks is true. *)
Definition scrutinee (is_int get_tag : variable) (x_is : Z) : flambda_type :=
  non_null_value
    (Variant (Some is_int) Or_unknown.Unknown (Some get_tag)
       Or_unknown.Unknown
       (Ext (Env_extension [(Name_var x, bottom Value)])
          (Env_extension [(Name_var x, this_tagged_immediate x_is)]))
       false).

Example variant_variables_has
  : has (Value_symbol a_block) (scrutinee zero zero 5).
Proof. yes. Qed.

Example variant_is_int_has_not
  : ~ has (Value_symbol a_block) (scrutinee one zero 5).
Proof. no. Qed.

Example variant_get_tag_has_not
  : ~ has (Value_symbol a_block) (scrutinee zero one 5).
Proof. no. Qed.

Example variant_extension_has_not
  : ~ has (Value_symbol a_block) (scrutinee zero zero 6).
Proof. no. Qed.

Example variant_when_immediate_has_not
  : ~ has (tagged 0) (scrutinee one zero 5).
Proof. no. Qed.

(* [Boxed_number]. *)
Example boxed_float_has
  : has (Value_symbol a_boxed_float)
      (box_number Boxable_naked_float
         (Type_naked_number Kind_naked_float (Ok (No_alias [42])))
         Alloc_mode.Heap).
Proof. yes. Qed.

Example boxed_float_contents_has_not
  : ~ has (Value_symbol a_boxed_float)
        (box_number Boxable_naked_float
           (Type_naked_number Kind_naked_float (Ok (No_alias [43])))
           Alloc_mode.Heap).
Proof. no. Qed.

Example boxed_float_kind_has_not
  : ~ has (Value_symbol a_boxed_float)
        (box_number Boxable_naked_int32
           (unknown (Naked_number Kind_naked_int32)) Alloc_mode.Heap).
Proof. no. Qed.

Example boxed_float_block_has_not
  : ~ has (Value_symbol a_block)
        (box_number Boxable_naked_float
           (unknown (Naked_number Kind_naked_float)) Alloc_mode.Heap).
Proof. no. Qed.

Example boxed_int64_has
  : has (Value_symbol a_boxed_int64)
      (box_number Boxable_naked_int64
         (Type_naked_number Kind_naked_int64 (Ok (No_alias [9])))
         Alloc_mode.Heap).
Proof. yes. Qed.

Example boxed_int64_contents_has_not
  : ~ has (Value_symbol a_boxed_int64)
        (box_number Boxable_naked_int64
           (Type_naked_number Kind_naked_int64 (Ok (No_alias [8])))
           Alloc_mode.Heap).
Proof. no. Qed.

Example boxed_int64_kind_has_not
  : ~ has (Value_symbol a_boxed_float)
        (box_number Boxable_naked_int64
           (unknown (Naked_number Kind_naked_int64)) Alloc_mode.Heap).
Proof. no. Qed.

(* [String]. *)
Example string_has
  : has (Value_symbol a_string) (this_immutable_string [104; 105]).
Proof. yes. Qed.

Example string_has_not
  : ~ has (Value_symbol a_string) (this_immutable_string [104]).
Proof. no. Qed.

(* [Closures]: the closure of the code [code] whose value slot holds a
   value of the type given. *)
Definition closure_type (code : code_id) (slot_type : flambda_type)
    : flambda_type :=
  exactly_this_closure unnamed_function_slot
    [(unnamed_function_slot, Or_unknown.Known (Function_type code))] []
    [(a_value_slot, slot_type)] Alloc_mode.Heap.

Example closure_has
  : has (Value_symbol a_closure)
      (closure_type some_code (this_tagged_immediate 3)).
Proof. yes. Qed.

Example closure_made_by_let_has
  : has (Value_closure some_code [(a_value_slot, tagged 3)])
      (closure_type some_code (this_tagged_immediate 3)).
Proof. yes. Qed.

Example closure_code_has_not
  : ~ has (Value_symbol a_closure)
        (closure_type other_code (this_tagged_immediate 3)).
Proof. no. Qed.

Example closure_value_slot_has_not
  : ~ has (Value_symbol a_closure)
        (closure_type some_code (this_tagged_immediate 4)).
Proof. no. Qed.

Example closure_block_has_not
  : ~ has (Value_symbol a_block)
        (closure_type some_code (this_tagged_immediate 3)).
Proof. no. Qed.

(* The closures of the set: the closure of the set at the slot of the
   closure is the closure itself, so its type has to hold of it. *)
Definition closure_with_closure_type (closure_type : flambda_type)
    : flambda_type :=
  exactly_this_closure unnamed_function_slot
    [(unnamed_function_slot, Or_unknown.Unknown)]
    [(unnamed_function_slot, closure_type)]
    [(a_value_slot, unknown Value)] Alloc_mode.Heap.

Example closure_types_has
  : has (Value_symbol a_closure)
      (closure_with_closure_type (alias_type_of Value (Symbol a_closure))).
Proof. yes. Qed.

Example closure_types_has_not
  : ~ has (Value_symbol a_closure)
        (closure_with_closure_type (alias_type_of Value (Symbol a_block))).
Proof. no. Qed.

(* Exactly and at least: the closure has one value slot, so it has not
   the type of a closure with exactly none, but has that of one with at
   least none; and it has not the type of a closure with at least a
   value slot that it lacks. *)
Example exactly_no_value_slot_has_not
  : ~ has (Value_symbol a_closure)
        (exactly_this_closure unnamed_function_slot
           [(unnamed_function_slot, Or_unknown.Unknown)] [] []
           Alloc_mode.Heap).
Proof. no. Qed.

Example at_least_no_value_slot_has
  : has (Value_symbol a_closure)
      (at_least_this_closure unnamed_function_slot
         [(unnamed_function_slot, Or_unknown.Unknown)] [] []
         Alloc_mode.Heap).
Proof. yes. Qed.

Example at_least_this_value_slot_has
  : has (Value_symbol a_closure)
      (at_least_this_closure unnamed_function_slot [] []
         [(a_value_slot, this_tagged_immediate 3)] Alloc_mode.Heap).
Proof. yes. Qed.

Example at_least_this_value_slot_type_has_not
  : ~ has (Value_symbol a_closure)
        (at_least_this_closure unnamed_function_slot [] []
           [(a_value_slot, this_tagged_immediate 4)] Alloc_mode.Heap).
Proof. no. Qed.

Example at_least_other_value_slot_has_not
  : ~ has (Value_symbol a_closure)
        (at_least_this_closure unnamed_function_slot [] []
           [(other_value_slot, unknown Value)] Alloc_mode.Heap).
Proof. no. Qed.

(* A function type of a function that the set has not says nothing. The
   type below is what the n-way join of the compiler made, before
   f489f9445e, of a set
   with the one function at the slot of the closure and of a set with
   a second function too: at least the first function, and the code of
   both (see the header of the Spec, on products). The closure of the
   first set has it, having no second function; and has it not if the
   code of the function that it has is another. *)
Definition join_of_closure_types (code : code_id) : flambda_type :=
  create_closures Alloc_mode.Heap
    (Row_like_for_closures
       [(unnamed_function_slot,
         Row_like_case
           (Closures_entry
              [(unnamed_function_slot,
                Or_unknown.Known (Function_type code));
               (other_function_slot,
                Or_unknown.Known (Function_type other_code))]
              [] [])
           (Row_like_index
              (Row_like_index_domain.At_least
                 (Set_of_closures_contents [unnamed_function_slot] []))
              tt)
           empty_env_extension)]).

Example closure_absent_function_has
  : has (Value_symbol a_closure) (join_of_closure_types some_code).
Proof. yes. Qed.

Example closure_absent_function_has_not
  : ~ has (Value_symbol a_closure) (join_of_closure_types other_code).
Proof. no. Qed.

(* Where the n-way join was not an upper bound, before f489f9445e
   (discrepancy T1-4). The first type below is
   what a projection of a value slot records of a closure: at least
   that value slot, and nothing of the code. Joined with the type of
   exactly a closure at the same slot with a known code, the n-way
   join gave the second: at least the value slot, and that code. The
   closure has the first and not the second, its code being another.
   Both joins now give a type that says nothing of the code. *)
Definition closure_of_unknown_code : flambda_type :=
  at_least_this_closure unnamed_function_slot [] []
    [(a_value_slot, unknown Value)] Alloc_mode.Heap.

Definition nway_join_with_known_code : flambda_type :=
  create_closures Alloc_mode.Heap
    (Row_like_for_closures
       [(unnamed_function_slot,
         Row_like_case
           (Closures_entry
              [(unnamed_function_slot,
                Or_unknown.Known (Function_type other_code))]
              [] [(a_value_slot, unknown Value)])
           (Row_like_index
              (Row_like_index_domain.At_least
                 (Set_of_closures_contents [] [a_value_slot]))
              tt)
           empty_env_extension)]).

Example nway_join_input_has
  : has (Value_symbol a_closure) closure_of_unknown_code.
Proof. yes. Qed.

Example nway_join_has_not
  : ~ has (Value_symbol a_closure) nway_join_with_known_code.
Proof. no. Qed.

(* A set of two functions: the closure at each of its function slots,
   made by a [Let]. The two differ in the slot only. *)
Definition first_function_slot : function_slot := 3%positive.

Definition two_functions : list (function_slot * code_id) :=
  [(first_function_slot, some_code); (other_function_slot, other_code)].

Definition closure_at (slot : function_slot) : value :=
  Value_closure_of_set slot two_functions [(a_value_slot, tagged 3)].

(* The closure at [slot] of a set of exactly those two functions and
   that value slot, whose closure at the other slot has the type
   given. *)
Definition closure_of_two_type (slot : function_slot)
    (other_closure_type : flambda_type) : flambda_type :=
  exactly_this_closure slot
    [(first_function_slot, Or_unknown.Known (Function_type some_code));
     (other_function_slot, Or_unknown.Known (Function_type other_code))]
    [(other_function_slot, other_closure_type)]
    [(a_value_slot, this_tagged_immediate 3)] Alloc_mode.Heap.

Example closure_of_two_has
  : has (closure_at first_function_slot)
      (closure_of_two_type first_function_slot (unknown Value)).
Proof. yes. Qed.

(* The closure at the other slot is another closure. *)
Example closure_at_other_slot_has_not
  : ~ has (closure_at other_function_slot)
        (closure_of_two_type first_function_slot (unknown Value)).
Proof. no. Qed.

(* The closure at the other slot of the set is that closure, which has
   the type of a closure at that slot and is no immediate. *)
Example closure_of_two_other_closure_has
  : has (closure_at first_function_slot)
      (closure_of_two_type first_function_slot
         (closure_of_two_type other_function_slot (unknown Value))).
Proof. yes. Qed.

Example closure_of_two_other_closure_has_not
  : ~ has (closure_at first_function_slot)
        (closure_of_two_type first_function_slot any_tagged_immediate).
Proof. no. Qed.

(* The code of each function of the set is what the type says. *)
Example closure_of_two_code_has_not
  : ~ has (closure_at first_function_slot)
        (exactly_this_closure first_function_slot
           [(first_function_slot, Or_unknown.Known (Function_type some_code));
            (other_function_slot,
             Or_unknown.Known (Function_type some_code))]
           []
           [(a_value_slot, this_tagged_immediate 3)] Alloc_mode.Heap).
Proof.
  unfold has. intros Hhas.
  apply exactly_this_closure_satisfied in Hhas
    as [functions [value_slots [Hclosure [_ [_ Hentry]]]]].
  compute in Hclosure. injection Hclosure as <- <-.
  apply closures_entry_lookup_iff in Hentry as [Hcode _].
  discriminate
    (Hcode other_function_slot (Function_type some_code) other_code eq_refl
       eq_refl).
Qed.

(* The same set, static: two symbols, each bound to the closure at one
   slot, with the symbols of both. The closure at the other slot of
   the set of the first is the second symbol. *)
Definition first_closure : symbol := 9%positive.
Definition other_closure : symbol := 10%positive.

Definition closure_symbols : list (function_slot * symbol) :=
  [(first_function_slot, first_closure); (other_function_slot, other_closure)].

Definition statics_with_set : static_env :=
  Static_env []
    [(first_closure,
      Static_closure_of_set first_function_slot closure_symbols two_functions
        [(a_value_slot, tagged 3)]);
     (other_closure,
      Static_closure_of_set other_function_slot closure_symbols two_functions
        [(a_value_slot, tagged 3)])].

Example static_closure_of_two_has
  : type_satisfied statics_with_set vars (Value_symbol first_closure)
      (closure_of_two_type first_function_slot
         (alias_type_of Value (Symbol other_closure))).
Proof. yes. Qed.

Example static_closure_of_two_has_not
  : ~ type_satisfied statics_with_set vars (Value_symbol first_closure)
        (closure_of_two_type first_function_slot
           (alias_type_of Value (Symbol first_closure))).
Proof. no. Qed.

(* [Mutable_block]: no value has it ([mutable_block_satisfied]), the
   machine having none; so there is no example of a value that has. *)

End Examples.

(* ======================================================================== *)
(* Proving with the meaning of types                                        *)
(* ======================================================================== *)

(* [simpl] and [cbn] leave [type_satisfied] and the functions defined with
   it alone from here on: rewrite with the equations above
   ([autorewrite with type_satisfied]) and with the statements. Unfolding
   still works where it is asked for, as the examples do with [compute]. *)
#[global] Arguments type_satisfied : simpl never.
#[global] Arguments head_of_kind_value_satisfied : simpl never.
#[global] Arguments head_of_kind_value_non_null_satisfied : simpl never.
#[global] Arguments row_like_for_blocks_satisfied : simpl never.
#[global] Arguments row_like_for_closures_satisfied : simpl never.
#[global] Arguments closures_entry_satisfied : simpl never.
#[global] Arguments env_extension_satisfied : simpl never.

(* ======================================================================== *)
(* When the static environment grows                                        *)
(* ======================================================================== *)

Section Statics_extend.

Variables statics statics' : static_env.
Hypothesis Hextend : statics_extend statics statics'.

Lemma block_of_extend (v : value) (block : Z * list value) :
  block_of statics v = Some block -> block_of statics' v = Some block.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    simpl; try discriminate; auto.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (proj2 Hextend s allocated Hlookup). auto.
Qed.

Lemma boxed_number_of_extend (v : value) (c : const) :
  boxed_number_of statics v = Some c -> boxed_number_of statics' v = Some c.
Proof.
  destruct v as [c' | code slots | slot decls slots | s | tag fields
    | elements | c' | l | pa_callee pa_args pa_arity];
    simpl; try discriminate; auto.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (proj2 Hextend s allocated Hlookup). auto.
Qed.

Lemma string_of_extend (v : value) (bytes : list Z) :
  string_of statics v = Some bytes -> string_of statics' v = Some bytes.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    simpl; try discriminate; auto.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (proj2 Hextend s allocated Hlookup). auto.
Qed.

Lemma closure_of_extend (v : value)
    (closure : code_id * list (value_slot * value)) :
  closure_of statics v = Some closure -> closure_of statics' v = Some closure.
Proof.
  destruct v; simpl; try discriminate; auto.
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ];
    rewrite (proj2 Hextend s allocated Hlookup); auto
  end.
Qed.

Lemma closure_in_set_extend (v : value)
    (closure :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value)) :
  closure_in_set statics v = Some closure ->
  closure_in_set statics' v = Some closure.
Proof.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    simpl; try discriminate; auto.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (proj2 Hextend s allocated Hlookup). auto.
Qed.

(* The closures of the set of a closure are the same ones. *)
Lemma project_function_slot_extend (v : value)
    (closure :
      function_slot * list (function_slot * code_id)
      * list (value_slot * value))
    (move_from move_to : function_slot) :
  closure_in_set statics v = Some closure ->
  project_function_slot statics' move_from move_to v
  = project_function_slot statics move_from move_to v.
Proof.
  unfold project_function_slot.
  destruct v as [c | code slots | slot decls slots | s | tag fields
    | elements | c | l | pa_callee pa_args pa_arity];
    simpl; try discriminate; try (intros _; reflexivity).
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
    [ | discriminate ].
  rewrite (proj2 Hextend s allocated Hlookup). reflexivity.
Qed.

Lemma relation_holds_extend (relation : Relation.t) (v : value) (n : Z) :
  relation_holds statics relation v n -> relation_holds statics' relation v n.
Proof.
  destruct relation; simpl.
  - auto.
  - intros [Himmediate | [[tag [fields Hblock]] Hn]];
      [ left; exact Himmediate | ].
    right. split; [ | exact Hn ]. exists tag, fields.
    exact (block_of_extend _ _ Hblock).
  - intros [Himmediate | [[fields Hblock] Htag]];
      [ left; exact Himmediate | ].
    right. split; [ | exact Htag ]. exists fields.
    exact (block_of_extend _ _ Hblock).
Qed.

(* What holds of the fields under one meaning of types holds under
   another that is implied by it for the types of the fields. *)
Lemma fields_satisfied_impl (P Q : value -> flambda_type -> Prop)
    (tys : list flambda_type) (fields : list value) :
  (∀ ty : flambda_type, ty ∈ tys -> ∀ v : value, P v ty -> Q v ty) ->
  fields_satisfied P tys fields -> fields_satisfied Q tys fields.
Proof.
  revert fields. induction tys as [ | ty tys IH]; intros fields Himpl.
  - auto.
  - destruct fields as [ | field fields]; simpl; [ auto | ].
    intros [Hfield Hfields]. split.
    + exact (Himpl ty (or_introl eq_refl) field Hfield).
    + apply IH; [ | exact Hfields ].
      intros ty' Hin. exact (Himpl ty' (or_intror Hin)).
Qed.

Lemma type_satisfied_statics_extend_all (vars : env) :
  (∀ (t : flambda_type) (v : value),
    type_satisfied statics vars v t -> type_satisfied statics' vars v t)
  ∧ (∀ (head : head_of_kind_value) (v : value),
      head_of_kind_value_satisfied statics vars v head ->
      head_of_kind_value_satisfied statics' vars v head)
  ∧ (∀ (head : head_of_kind_value_non_null) (v : value),
      head_of_kind_value_non_null_satisfied statics vars v head ->
      head_of_kind_value_non_null_satisfied statics' vars v head)
  ∧ (∀ (blocks : row_like_for_blocks) (tag : Z) (fields : list value),
      row_like_for_blocks_satisfied statics vars tag fields blocks ->
      row_like_for_blocks_satisfied statics' vars tag fields blocks)
  ∧ (∀ (closures : row_like_for_closures) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closure_in_set statics v = Some (slot, functions, value_slots) ->
      row_like_for_closures_satisfied statics vars v slot functions
        value_slots closures ->
      row_like_for_closures_satisfied statics' vars v slot functions
        value_slots closures)
  ∧ (∀ (entry : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closure_in_set statics v = Some (slot, functions, value_slots) ->
      closures_entry_satisfied statics vars v slot functions value_slots
        entry ->
      closures_entry_satisfied statics' vars v slot functions value_slots
        entry)
  ∧ (∀ extension : env_extension,
      env_extension_satisfied statics vars extension ->
      env_extension_satisfied statics' vars extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros [ | [head | s] | ] Hhead v; autorewrite with type_satisfied;
      simpl; auto.
  - (* Type_naked_immediate *)
    intros [ | [head | s] | ] v; autorewrite with type_satisfied; simpl; auto.
    intros [i [Hv [Hnumbers Hrelations]]]. exists i.
    split; [ exact Hv | ]. split; [ exact Hnumbers | ].
    intros relation n Hin. destruct (Hrelations relation n Hin) as [w [Hw H]].
    exists w. split; [ exact Hw | exact (relation_holds_extend _ _ _ H) ].
  - intros k ty v. autorewrite with type_satisfied. auto.
  - intros ty v. autorewrite with type_satisfied. auto.
  - intros ty v. autorewrite with type_satisfied. auto.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null v. autorewrite with type_satisfied.
    intros [Hnull | [Hnot_null [Hhead His_null]]]; [ left; exact Hnull | ].
    right. split; [ exact Hnot_null | ]. split; [ | exact His_null ].
    destruct non_null as [ | head | ]; auto.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions v.
    autorewrite with type_satisfied.
    intros [[i [Hv [His_int [Himmediate Hextension]]]]
           | [tag [fields [Hblock [His_int [Hget_tag [Hrow Hextension]]]]]]].
    + left. exists i. split; [ exact Hv | ]. split; [ exact His_int | ].
      split.
      * destruct immediates as [ty | ]; auto.
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        exact (proj1 Hextensions Hextension).
    + right. exists tag, fields.
      split; [ exact (block_of_extend _ _ Hblock) | ].
      split; [ exact His_int | ]. split; [ exact Hget_tag | ]. split.
      * destruct blocks as [blocks | ]; auto.
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        exact (proj2 Hextensions Hextension).
  - (* Mutable_block *)
    intros alloc_mode v. autorewrite with type_satisfied. auto.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents v. autorewrite with type_satisfied.
    intros [c [Hboxed [Hkind Hc]]]. exists c.
    split; [ exact (boxed_number_of_extend _ _ Hboxed) | ]. auto.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures v.
    autorewrite with type_satisfied.
    intros [slot [functions [value_slots [Hclosure Hrow]]]].
    exists slot, functions, value_slots.
    split; [ exact (closure_in_set_extend _ _ Hclosure) | ].
    exact (Hclosures v slot functions value_slots Hclosure Hrow).
  - (* String *)
    intros strs v. autorewrite with type_satisfied.
    intros [bytes [Hstring Hin]]. exists bytes.
    split; [ exact (string_of_extend _ _ Hstring) | exact Hin ].
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother tag fields.
    rewrite !row_like_for_blocks_find_iff.
    assert (Hcase :
      ∀ case : row_like_block_case,
        P_block_case
          (fun t =>
             ∀ v : value,
               type_satisfied statics vars v t ->
               type_satisfied statics' vars v t)
          (fun extension =>
             env_extension_satisfied statics vars extension ->
             env_extension_satisfied statics' vars extension)
          case ->
        row_like_block_case_satisfied statics vars tag fields case ->
        row_like_block_case_satisfied statics' vars tag fields case).
    { intros case [Htypes Hextension] [Hindex [Hfields Hext]].
      split; [ exact Hindex | ]. split; [ | exact (Hextension Hext) ].
      exact (fields_satisfied_impl _ _ _ _ Htypes Hfields). }
    destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ] eqn:Hfind.
    + apply Hcase.
      exact (Hknown tag case (keyed_lookup_in Z.eqb Z.eqb_eq _ _ _ Hfind)).
    + auto.
    + destruct other_tags as [case | ]; [ exact (Hcase case Hother) | auto ].
  - (* Row_like_for_closures *)
    intros known_closures Hknown v slot functions value_slots Hclosure.
    rewrite !row_like_for_closures_lookup_iff.
    intros [case [Hlookup [Hindex [Hentry Hextension]]]]. exists case.
    split; [ exact Hlookup | ].
    destruct (Hknown slot case (lookup_in _ _ _ Hlookup))
      as [IHentry IHextension].
    split; [ exact Hindex | ]. split; [ | exact (IHextension Hextension) ].
    exact (IHentry v slot functions value_slots Hclosure Hentry).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots v
      slot functions value_slots Hclosure.
    rewrite !closures_entry_lookup_iff.
    intros [Hfunctions [Hclosure_types Hvalue_slots]].
    split; [ exact Hfunctions | ]. split.
    + intros g ty closure Hlookup Hproject.
      rewrite (project_function_slot_extend _ _ _ _ Hclosure) in Hproject.
      apply (Hclosures g ty (lookup_in _ _ _ Hlookup)).
      exact (Hclosure_types g ty closure Hlookup Hproject).
    + intros w ty contents Hlookup Hcontents.
      apply (Hslots w ty (lookup_in _ _ _ Hlookup)).
      exact (Hvalue_slots w ty contents Hlookup Hcontents).
  - (* Env_extension *)
    intros equations Htypes.
    rewrite !env_extension_lookup_iff.
    intros Hequations n ty Hlookup.
    destruct (Hequations n ty Hlookup) as [v [Hv Hty]]. exists v.
    split; [ exact Hv | ].
    exact
      (Htypes n ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup) v Hty).
Qed.

End Statics_extend.

Theorem type_satisfied_statics_extend
  : type_satisfied_statics_extend_statement.
Proof.
  intros statics statics' vars v t Hextend.
  exact (proj1 (type_satisfied_statics_extend_all _ _ Hextend vars) t v).
Qed.

Theorem env_extension_satisfied_statics_extend
  : env_extension_satisfied_statics_extend_statement.
Proof.
  intros statics statics' vars extension Hextend.
  destruct (type_satisfied_statics_extend_all _ _ Hextend vars)
    as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
  exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* The variables that the meaning of a type depends on                      *)
(* ======================================================================== *)

(* An environment keeps the values that another gives to some
   variables: it may give a value to more of them. The meaning of a
   type reads the environment only to find a value in it, so what
   holds under the one holds under the other. *)
Definition keeps (vars vars' : env) (xs : list variable) : Prop :=
  ∀ (x : variable) (w : value),
    x ∈ xs -> lookup x vars = Some w -> lookup x vars' = Some w.

Lemma keeps_app (vars vars' : env) (xs ys : list variable) :
  keeps vars vars' (xs ++ ys) <-> keeps vars vars' xs ∧ keeps vars vars' ys.
Proof.
  unfold keeps. split.
  - intros Hkeeps. split; intros x w Hin; apply Hkeeps, in_or_app; auto.
  - intros [Hxs Hys] x w Hin. apply in_app_or in Hin as [Hin | Hin]; auto.
Qed.

Lemma keeps_flat_map {A : Type} (vars vars' : env) (f : A -> list variable)
    (l : list A) (a : A) :
  keeps vars vars' (flat_map f l) -> a ∈ l -> keeps vars vars' (f a).
Proof.
  intros Hkeeps Hin x w Hx. apply Hkeeps, in_flat_map. exists a. auto.
Qed.

Lemma keeps_eval_simple (vars vars' : env) (s : simple) (v : value) :
  keeps vars vars' (free_variables_of_simple s) ->
  eval_simple vars s = Some v -> eval_simple vars' s = Some v.
Proof.
  destruct s as [x | sym | c]; simpl; [ | auto | auto ].
  intros Hkeeps. apply Hkeeps. left. reflexivity.
Qed.

Lemma keeps_variable_is (vars vars' : env) (x : option variable) (n : Z) :
  keeps vars vars' (free_variables_of_option x) ->
  variable_is vars x n -> variable_is vars' x n.
Proof.
  destruct x as [x | ]; simpl; [ | auto ].
  intros Hkeeps. apply Hkeeps. left. reflexivity.
Qed.

Lemma keeps_descr {head : Type} (vars vars' : env) (k : flambda_kind)
    (free_variables_of_head : head -> list variable)
    (head_satisfied head_satisfied' : head -> Prop) (v : value)
    (ty : type_descr head) :
  keeps vars vars' (free_variables_of_descr free_variables_of_head ty) ->
  (∀ h : head,
    ty = Ok (No_alias h) -> head_satisfied h -> head_satisfied' h) ->
  descr_satisfied vars k head_satisfied v ty ->
  descr_satisfied vars' k head_satisfied' v ty.
Proof.
  destruct ty as [ | [h | s] | ]; simpl; [ auto | | | auto ].
  - intros _ Hhead. exact (Hhead h eq_refl).
  - intros Hkeeps _ [Hs Hkind].
    split; [ exact (keeps_eval_simple _ _ _ _ Hkeeps Hs) | exact Hkind ].
Qed.

Lemma type_satisfied_monotone_all (statics : static_env) :
  (∀ (t : flambda_type) (vars vars' : env) (v : value),
    keeps vars vars' (free_variables t) ->
    type_satisfied statics vars v t -> type_satisfied statics vars' v t)
  ∧ (∀ (head : head_of_kind_value) (vars vars' : env) (v : value),
      keeps vars vars' (free_variables_of_head_of_kind_value head) ->
      head_of_kind_value_satisfied statics vars v head ->
      head_of_kind_value_satisfied statics vars' v head)
  ∧ (∀ (head : head_of_kind_value_non_null) (vars vars' : env) (v : value),
      keeps vars vars' (free_variables_of_head_of_kind_value_non_null head) ->
      head_of_kind_value_non_null_satisfied statics vars v head ->
      head_of_kind_value_non_null_satisfied statics vars' v head)
  ∧ (∀ (blocks : row_like_for_blocks) (vars vars' : env) (tag : Z)
       (fields : list value),
      keeps vars vars' (free_variables_of_row_like_for_blocks blocks) ->
      row_like_for_blocks_satisfied statics vars tag fields blocks ->
      row_like_for_blocks_satisfied statics vars' tag fields blocks)
  ∧ (∀ (closures : row_like_for_closures) (vars vars' : env) (v : value)
       (slot : function_slot) (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      keeps vars vars' (free_variables_of_row_like_for_closures closures) ->
      row_like_for_closures_satisfied statics vars v slot functions
        value_slots closures ->
      row_like_for_closures_satisfied statics vars' v slot functions
        value_slots closures)
  ∧ (∀ (entry : closures_entry) (vars vars' : env) (v : value)
       (slot : function_slot) (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      keeps vars vars' (free_variables_of_closures_entry entry) ->
      closures_entry_satisfied statics vars v slot functions value_slots
        entry ->
      closures_entry_satisfied statics vars' v slot functions value_slots
        entry)
  ∧ (∀ (extension : env_extension) (vars vars' : env),
      keeps vars vars' (free_variables_of_env_extension extension) ->
      env_extension_satisfied statics vars extension ->
      env_extension_satisfied statics vars' extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead vars vars' v Hkeeps. autorewrite with type_satisfied.
    apply (keeps_descr _ _ _ _ _ _ _ _ Hkeeps).
    intros h ->. exact (Hhead vars vars' v Hkeeps).
  - (* Type_naked_immediate *)
    intros ty vars vars' v Hkeeps. autorewrite with type_satisfied.
    apply (keeps_descr _ _ _ _ _ _ _ _ Hkeeps).
    intros head ->. simpl in Hkeeps.
    intros [i [Hv [Hnumbers Hrelations]]]. exists i.
    split; [ exact Hv | ]. split; [ exact Hnumbers | ].
    intros relation n Hin. destruct (Hrelations relation n Hin) as [w [Hw H]].
    exists w. split; [ | exact H ]. unfold value_of_name in *.
    apply (keeps_eval_simple vars vars' _ w); [ | exact Hw ].
    exact (keeps_flat_map _ _ _ _ (relation, n) Hkeeps Hin).
  - intros k ty vars vars' v Hkeeps. autorewrite with type_satisfied.
    apply (keeps_descr _ _ _ _ _ _ _ _ Hkeeps). auto.
  - intros ty vars vars' v Hkeeps. autorewrite with type_satisfied.
    apply (keeps_descr _ _ _ _ _ _ _ _ Hkeeps). auto.
  - intros ty vars vars' v Hkeeps. autorewrite with type_satisfied.
    apply (keeps_descr _ _ _ _ _ _ _ _ Hkeeps). auto.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null vars vars' v Hkeeps.
    simpl in Hkeeps. apply keeps_app in Hkeeps as [Hkeeps_head Hkeeps_null].
    autorewrite with type_satisfied.
    intros [[Hnull His_null] | [Hnot_null [Hhead His_null]]].
    + left. split; [ exact Hnull | ].
      destruct is_null as [ | is_null]; [ exact His_null | ].
      exact (keeps_variable_is _ _ _ _ Hkeeps_null His_null).
    + right. split; [ exact Hnot_null | ]. split.
      * destruct non_null as [ | head | ]; auto.
        exact (Hnon_null vars vars' v Hkeeps_head Hhead).
      * destruct is_null as [ | is_null]; [ exact His_null | ].
        exact (keeps_variable_is _ _ _ _ Hkeeps_null His_null).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions vars vars' v Hkeeps.
    simpl in Hkeeps.
    apply keeps_app in Hkeeps as [Hkeeps_is_int Hkeeps].
    apply keeps_app in Hkeeps as [Hkeeps_immediates Hkeeps].
    apply keeps_app in Hkeeps as [Hkeeps_get_tag Hkeeps].
    apply keeps_app in Hkeeps as [Hkeeps_blocks Hkeeps_extensions].
    autorewrite with type_satisfied.
    intros [[i [Hv [His_int [Himmediate Hextension]]]]
           | [tag [fields [Hblock [His_int [Hget_tag [Hrow Hextension]]]]]]].
    + left. exists i. split; [ exact Hv | ].
      split; [ exact (keeps_variable_is _ _ _ _ Hkeeps_is_int His_int) | ].
      split.
      * destruct immediates as [ty | ]; [ | exact I ].
        exact (Himmediates vars vars' _ Hkeeps_immediates Himmediate).
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        apply keeps_app in Hkeeps_extensions as [Hkeeps_immediate _].
        exact (proj1 Hextensions vars vars' Hkeeps_immediate Hextension).
    + right. exists tag, fields. split; [ exact Hblock | ].
      split; [ exact (keeps_variable_is _ _ _ _ Hkeeps_is_int His_int) | ].
      split; [ exact (keeps_variable_is _ _ _ _ Hkeeps_get_tag Hget_tag) | ].
      split.
      * destruct blocks as [blocks | ]; [ | exact I ].
        exact (Hblocks vars vars' tag fields Hkeeps_blocks Hrow).
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        apply keeps_app in Hkeeps_extensions as [_ Hkeeps_block].
        exact (proj2 Hextensions vars vars' Hkeeps_block Hextension).
  - (* Mutable_block *)
    intros alloc_mode vars vars' v _. autorewrite with type_satisfied. auto.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents vars vars' v Hkeeps.
    autorewrite with type_satisfied.
    intros [c [Hboxed [Hkind Hc]]]. exists c.
    split; [ exact Hboxed | ]. split; [ exact Hkind | ].
    exact (Hcontents vars vars' _ Hkeeps Hc).
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures vars vars' v Hkeeps.
    autorewrite with type_satisfied.
    intros [slot [functions [value_slots [Hclosure Hrow]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    exact (Hclosures vars vars' v slot functions value_slots Hkeeps Hrow).
  - (* String *)
    intros strs vars vars' v _. autorewrite with type_satisfied. auto.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother vars vars' tag
      fields Hkeeps.
    simpl in Hkeeps. apply keeps_app in Hkeeps as [Hkeeps_known Hkeeps_other].
    rewrite !row_like_for_blocks_find_iff.
    assert (Hcase :
      ∀ case : row_like_block_case,
        P_block_case
          (fun t =>
             ∀ (vars vars' : env) (v : value),
               keeps vars vars' (free_variables t) ->
               type_satisfied statics vars v t ->
               type_satisfied statics vars' v t)
          (fun extension =>
             ∀ vars vars' : env,
               keeps vars vars' (free_variables_of_env_extension extension) ->
               env_extension_satisfied statics vars extension ->
               env_extension_satisfied statics vars' extension)
          case ->
        keeps vars vars'
          (free_variables_of_block_case free_variables
             free_variables_of_env_extension case) ->
        row_like_block_case_satisfied statics vars tag fields case ->
        row_like_block_case_satisfied statics vars' tag fields case).
    { intros case [Htypes Hextension] Hkeeps [Hindex [Hfields Hext]].
      apply keeps_app in Hkeeps as [Hkeeps_fields Hkeeps_extension].
      split; [ exact Hindex | ].
      split; [ | exact (Hextension vars vars' Hkeeps_extension Hext) ].
      apply (fields_satisfied_impl _ _ _ _) with (2 := Hfields).
      intros ty Hin v.
      exact (Htypes ty Hin vars vars' v
               (keeps_flat_map _ _ _ _ ty Hkeeps_fields Hin)). }
    destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ] eqn:Hfind.
    + apply (keyed_lookup_in Z.eqb Z.eqb_eq) in Hfind.
      apply (Hcase case (Hknown tag case Hfind)).
      exact (keeps_flat_map _ _ _ _ _ Hkeeps_known Hfind).
    + auto.
    + destruct other_tags as [case | ]; [ | auto ].
      exact (Hcase case Hother Hkeeps_other).
  - (* Row_like_for_closures *)
    intros known_closures Hknown vars vars' v slot functions value_slots
      Hkeeps.
    rewrite !row_like_for_closures_lookup_iff.
    intros [case [Hlookup [Hindex [Hentry Hextension]]]]. exists case.
    split; [ exact Hlookup | ]. apply lookup_in in Hlookup.
    destruct (Hknown slot case Hlookup) as [IHentry IHextension].
    simpl in Hkeeps.
    apply (keeps_flat_map _ _ _ _ _ Hkeeps) in Hlookup. simpl in Hlookup.
    apply keeps_app in Hlookup as [Hkeeps_entry Hkeeps_extension].
    split; [ exact Hindex | ].
    split; [ | exact (IHextension vars vars' Hkeeps_extension Hextension) ].
    exact (IHentry vars vars' v slot functions value_slots Hkeeps_entry
             Hentry).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      vars vars' v slot functions value_slots Hkeeps.
    simpl in Hkeeps.
    apply keeps_app in Hkeeps as [Hkeeps_closures Hkeeps_slots].
    rewrite !closures_entry_lookup_iff.
    intros [Hfunctions [Hclosure_types Hvalue_slots]].
    split; [ exact Hfunctions | ]. split.
    + intros g ty closure Hlookup Hproject.
      pose proof (lookup_in _ _ _ Hlookup) as Hin.
      apply (Hclosures g ty Hin vars vars').
      * exact (keeps_flat_map _ _ _ _ _ Hkeeps_closures Hin).
      * exact (Hclosure_types g ty closure Hlookup Hproject).
    + intros w ty contents Hlookup Hcontents.
      pose proof (lookup_in _ _ _ Hlookup) as Hin.
      apply (Hslots w ty Hin vars vars').
      * exact (keeps_flat_map _ _ _ _ _ Hkeeps_slots Hin).
      * exact (Hvalue_slots w ty contents Hlookup Hcontents).
  - (* Env_extension *)
    intros equations Htypes vars vars' Hkeeps.
    simpl in Hkeeps.
    rewrite !env_extension_lookup_iff.
    intros Hequations n ty Hlookup.
    destruct (Hequations n ty Hlookup) as [v [Hv Hty]]. exists v.
    pose proof (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hlookup) as Hin.
    pose proof (keeps_flat_map _ _ _ _ _ Hkeeps Hin) as Hkeeps_binding.
    simpl in Hkeeps_binding.
    apply keeps_app in Hkeeps_binding as [Hkeeps_name Hkeeps_ty]. split.
    + exact (keeps_eval_simple vars vars' _ v Hkeeps_name Hv).
    + exact (Htypes n ty Hin vars vars' v Hkeeps_ty Hty).
Qed.

Theorem type_satisfied_monotone : type_satisfied_monotone_statement.
Proof.
  intros statics vars vars' v t Hkeeps.
  exact (proj1 (type_satisfied_monotone_all statics) t vars vars' v Hkeeps).
Qed.

Theorem env_extension_satisfied_monotone
  : env_extension_satisfied_monotone_statement.
Proof.
  intros statics vars vars' extension Hkeeps.
  destruct (type_satisfied_monotone_all statics)
    as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
  exact (Hextension extension vars vars' Hkeeps).
Qed.

(* Two environments that give the same values to some variables keep
   each other's. *)
Lemma agree_keeps (vars vars' : env) (xs : list variable) :
  (∀ x : variable, x ∈ xs -> lookup x vars = lookup x vars') ->
  keeps vars vars' xs ∧ keeps vars' vars xs.
Proof.
  intros Hagree.
  split; intros x w Hin Hlookup;
    [ rewrite <- (Hagree x Hin) | rewrite (Hagree x Hin) ]; exact Hlookup.
Qed.

Theorem type_satisfied_coincide : type_satisfied_coincide_statement.
Proof.
  intros statics vars vars' v t Hagree.
  destruct (agree_keeps _ _ _ Hagree) as [Hkeeps Hkeeps'].
  split; apply type_satisfied_monotone; assumption.
Qed.

Theorem env_extension_satisfied_coincide
  : env_extension_satisfied_coincide_statement.
Proof.
  intros statics vars vars' extension Hagree.
  destruct (agree_keeps _ _ _ Hagree) as [Hkeeps Hkeeps'].
  split; apply env_extension_satisfied_monotone; assumption.
Qed.

Theorem type_satisfied_bind : type_satisfied_bind_statement.
Proof.
  intros statics vars x w v t Hfresh. apply type_satisfied_coincide.
  intros y Hin. simpl.
  destruct (Pos.eqb_spec y x) as [-> | _]; [ contradiction | reflexivity ].
Qed.

Theorem unknown_is_top : unknown_is_top_statement.
Proof.
  intros statics vars v t Hsatisfied. apply unknown_satisfied.
  exact (type_satisfied_kind _ _ _ _ Hsatisfied).
Qed.

(* ======================================================================== *)
(* Types with no value, and heads with no value in common                   *)
(* ======================================================================== *)

Theorem is_obviously_bottom_satisfied
  : is_obviously_bottom_satisfied_statement.
Proof.
  intros statics vars v t Hbottom.
  destruct t as [ty | ty | k ty | ty | ty]; autorewrite with type_satisfied;
    destruct ty as [ | d | ]; try discriminate; intros [].
Qed.

Theorem is_obviously_unknown_satisfied
  : is_obviously_unknown_satisfied_statement.
Proof.
  intros statics vars v t Hunknown.
  destruct t as [ty | ty | k ty | ty | ty]; autorewrite with type_satisfied;
    destruct ty as [ | d | ]; try discriminate; reflexivity.
Qed.

Theorem row_like_for_blocks_is_bottom_satisfied
  : row_like_for_blocks_is_bottom_satisfied_statement.
Proof.
  intros statics vars tag fields [known_tags other_tags alloc_mode] Hbottom.
  rewrite row_like_for_blocks_find_iff.
  destruct known_tags as [ | case known_tags]; [ | discriminate ].
  destruct other_tags as [case | ]; [ discriminate | ]. simpl. auto.
Qed.

Lemma unboxed_kind_injective (k k' : boxable_number) :
  unboxed_kind k = unboxed_kind k' -> k = k'.
Proof. destruct k, k'; try discriminate; reflexivity. Qed.

Theorem heads_compatible_satisfied : heads_compatible_satisfied_statement.
Proof.
  intros statics vars v head head' Hhead Hhead'.
  pose proof (observations_disjoint statics v) as [Hblock [Hclosure Hboxed]].
  assert (Hclosure_in_set :
    ∀ closure, closure_in_set statics v = Some closure ->
               closure_of statics v <> None).
  { intros [[slot decls] slots] Hin.
    pose proof (closure_in_set_slot _ _ _ _ _ Hin) as Hbound.
    unfold is_bound in Hbound.
    destruct (lookup slot decls) as [code | ] eqn:Hcode;
      [ | discriminate Hbound ].
    rewrite
      (proj2 (closure_in_set_of statics v code slots)
         (ex_intro _ slot (ex_intro _ decls (conj Hin Hcode)))).
    discriminate. }
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | alloc_mode | k contents alloc_mode
                   | by_function_slot alloc_mode | strs];
    destruct head' as [is_int' immediates' get_tag' blocks' extensions'
                         is_unique'
                      | alloc_mode' | k' contents' alloc_mode'
                      | by_function_slot' alloc_mode' | strs'];
    autorewrite with type_satisfied in Hhead, Hhead'; simpl; auto;
    repeat
      match goal with
      | H : False |- _ => destruct H
      | H : _ ∨ _ |- _ => destruct H
      | H : ∃ _, _ |- _ => destruct H
      | H : _ ∧ _ |- _ => destruct H
      end;
    subst; try discriminate;
    repeat
      match goal with
      | H : block_of statics v = Some _ |- _ =>
        assert (block_of statics v <> None) by (rewrite H; discriminate);
        clear H
      | H : boxed_number_of statics v = Some ?c,
        H' : boxed_number_of statics v = Some ?c' |- _ =>
        rewrite H in H'; injection H' as <-
      | H : kind_of_const ?c = unboxed_kind ?k,
        H' : kind_of_const ?c = unboxed_kind ?k' |- _ =>
        rewrite H in H'; apply unboxed_kind_injective in H'; clear H
      | H : boxed_number_of statics v = Some _ |- _ =>
        assert (boxed_number_of statics v <> None)
          by (rewrite H; discriminate);
        clear H
      | H : string_of statics v = Some _ |- _ =>
        assert (string_of statics v <> None) by (rewrite H; discriminate);
        clear H
      | H : closure_in_set statics v = Some _ |- _ =>
        apply Hclosure_in_set in H
      end;
    try congruence; try tauto.
Qed.

(* ======================================================================== *)
(* What a relation says of the value it relates                             *)
(* ======================================================================== *)

Lemma lookup_unknown_cases {A : Type} (tag : Z) (tags : list Z) :
  Keyed.lookup Z.eqb tag
    (map (fun tag => (tag, @Or_unknown.Unknown A)) tags)
  = if existsb (Z.eqb tag) tags then Some Or_unknown.Unknown else None.
Proof.
  induction tags as [ | tag' tags IH]; simpl; [ reflexivity | ].
  destruct (Z.eqb tag tag'); [ reflexivity | exact IH ].
Qed.

Theorem blocks_with_these_tags_satisfied
  : blocks_with_these_tags_satisfied_statement.
Proof.
  intros statics vars v tags alloc_mode. unfold blocks_with_these_tags.
  rewrite (create_variant_satisfied statics vars v).
  assert (Hrow :
    ∀ (tag : Z) (fields : list value),
      row_like_for_blocks_satisfied statics vars tag fields
        (Row_like_for_blocks
           (map (fun tag => (tag, Or_unknown.Unknown)) tags)
           Or_bottom.Bottom alloc_mode)
      <-> tag ∈ tags).
  { intros tag fields. rewrite row_like_for_blocks_find_iff.
    rewrite lookup_unknown_cases.
    destruct (existsb (Z.eqb tag) tags) eqn:Hamong.
    - apply existsb_exists in Hamong as [tag' [Hin Heq]].
      apply Z.eqb_eq in Heq as <-. tauto.
    - split; [ intros [] | ]. intros Hin.
      assert (existsb (Z.eqb tag) tags = true); [ | congruence ].
      apply existsb_exists. exists tag. split; [ exact Hin | ].
      apply Z.eqb_refl. }
  split.
  - intros [[i [_ [Hbottom _]]] | [tag [fields [Hblock [Hblocks _]]]]].
    + destruct (bottom_satisfied _ _ _ _ Hbottom).
    + exists tag, fields. split; [ exact Hblock | ].
      exact (proj1 (Hrow tag fields) Hblocks).
  - intros [tag [fields [Hblock Hin]]]. right. exists tag, fields.
    split; [ exact Hblock | ]. split; [ | exact I ].
    exact (proj2 (Hrow tag fields) Hin).
Qed.

Theorem relation_is_null : relation_is_null_statement.
Proof.
  intros statics vars w n [[-> ->] | [Hkind [Hnot_null ->]]].
  - left. split; [ reflexivity | ]. apply null_satisfied. reflexivity.
  - right. split; [ reflexivity | ]. apply any_non_null_value_satisfied.
    split; assumption.
Qed.

Theorem relation_is_int : relation_is_int_statement.
Proof.
  intros statics vars w n [[Himmediate ->] | [Hblock ->]].
  - left. split; [ reflexivity | ].
    apply any_tagged_immediate_satisfied. exact Himmediate.
  - right. split; [ reflexivity | ].
    apply any_block_satisfied. exact Hblock.
Qed.

Theorem relation_get_tag : relation_get_tag_statement.
Proof.
  intros statics vars w n tags alloc_mode
    [Himmediate | [[fields Hblock] Htag]];
    [ left; exact Himmediate | ].
  right. split; [ exact Htag | ]. intros Hin.
  apply blocks_with_these_tags_satisfied. exists n, fields.
  split; assumption.
Qed.

(* ======================================================================== *)
(* What the operations on types rely on                                     *)
(* ======================================================================== *)

Theorem block_case_empty : block_case_empty_statement.
Proof.
  intros statics vars tag fields case ty Hwidth Hin Hempty
    [[Hsize _] [Hfields _]].
  apply In_nth_error in Hin as [i Hi].
  assert (Hlength : length (case_maps_to case) <= length fields).
  { unfold block_index_size in Hwidth.
    destruct (index_domain (case_index case)); lia. }
  assert (Hi' : i < length (case_maps_to case)).
  { apply nth_error_Some. rewrite Hi. discriminate. }
  destruct (nth_error fields i) as [field | ] eqn:Hfield.
  - apply (Hempty field).
    exact
      (proj1 (fields_satisfied_nth_iff _ _ _) Hfields i ty field Hi Hfield).
  - apply nth_error_None in Hfield. lia.
Qed.

Theorem closures_case_value_slot : closures_case_value_slot_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension w ty
    [Hindex [Hentry _]] Hamong Hty.
  simpl in Hindex, Hentry.
  apply closures_entry_lookup_iff in Hentry as [_ [_ Hslots]].
  assert (Hbound : is_bound w value_slots = true).
  { rewrite <- among_domain. unfold closures_index_satisfied in Hindex.
    unfold closures_index_contents in Hamong.
    destruct (index_domain index) as [contents | contents].
    - destruct Hindex as [_ Hsame]. rewrite <- (Hsame w). exact Hamong.
    - destruct Hindex as [_ Hall].
      exact (proj1 (all_among _ _) Hall w Hamong). }
  unfold is_bound in Hbound.
  destruct (lookup w value_slots) as [contents | ] eqn:Hcontents;
    [ | discriminate Hbound ].
  exists contents. split; [ reflexivity | ].
  exact (Hslots w ty contents Hty Hcontents).
Qed.

(* The kinds of the fields of a mixed record give its shape back, and
   are not those of a block of values if it has a flat suffix. *)
Lemma map_naked_number_inj (kinds kinds' : list naked_number_kind) :
  map Naked_number kinds = map Naked_number kinds' -> kinds = kinds'.
Proof.
  revert kinds'. induction kinds as [ | k kinds IH];
    intros [ | k' kinds']; simpl; try discriminate; [ reflexivity | ].
  intros [= -> Hkinds]. rewrite (IH kinds' Hkinds). reflexivity.
Qed.

Lemma mixed_kinds_inj (prefix prefix' : nat)
    (suffix suffix' : list naked_number_kind) :
  repeat Value prefix ++ map Naked_number suffix
  = repeat Value prefix' ++ map Naked_number suffix' ->
  prefix = prefix' ∧ suffix = suffix'.
Proof.
  revert prefix'. induction prefix as [ | prefix IH];
    intros [ | prefix']; simpl.
  - intros Hsuffix. split; [ reflexivity | ].
    exact (map_naked_number_inj _ _ Hsuffix).
  - destruct suffix; discriminate.
  - destruct suffix'; discriminate.
  - intros [= Hkinds]. destruct (IH prefix' Hkinds) as [-> ->].
    split; reflexivity.
Qed.

Lemma values_not_mixed (size prefix : nat) (suffix : list naked_number_kind) :
  repeat Value size = repeat Value prefix ++ map Naked_number suffix ->
  suffix = [].
Proof.
  revert prefix. induction size as [ | size IH]; intros [ | prefix]; simpl.
  - destruct suffix; [ reflexivity | discriminate ].
  - discriminate.
  - destruct suffix; [ reflexivity | discriminate ].
  - intros [= Hkinds]. exact (IH prefix Hkinds).
Qed.

Theorem block_shapes_disjoint : block_shapes_disjoint_statement.
Proof.
  intros shape shape' tag fields Hcanonical Hcanonical' [Htag Hkinds]
    [Htag' Hkinds'].
  unfold no_scan_tag, double_array_tag in *.
  destruct shape as [[ | prefix suffix] | ],
    shape' as [[ | prefix' suffix'] | ];
    simpl in *; try reflexivity; try lia.
  - rewrite Hkinds in Hkinds'.
    destruct (Hcanonical' (values_not_mixed _ _ _ Hkinds')).
  - rewrite Hkinds' in Hkinds.
    destruct (Hcanonical (values_not_mixed _ _ _ Hkinds)).
  - rewrite Hkinds in Hkinds'.
    destruct (mixed_kinds_inj _ _ _ _ Hkinds') as [-> ->]. reflexivity.
Qed.

(* ======================================================================== *)
(* What meet and the joins read of a type                                   *)
(* ======================================================================== *)

Theorem heads_of_meet : heads_of_meet_statement.
Proof. split; [ reflexivity | intros imm; reflexivity ]. Qed.

Theorem block_shape_element_kind_field
  : block_shape_element_kind_field_statement.
Proof.
  intros shape tag fields index field k [_ Hkinds] Hfield Hkind.
  apply (map_nth_error kind_of_value) in Hfield. rewrite Hkinds in Hfield.
  destruct shape as [[ | prefix suffix] | ]; simpl in Hkind, Hfield.
  - injection Hkind as <-. apply nth_error_In, repeat_spec in Hfield.
    exact Hfield.
  - destruct (Nat.ltb index prefix) eqn:Hlt.
    + injection Hkind as <-. apply Nat.ltb_lt in Hlt.
      rewrite nth_error_app1 in Hfield by (rewrite repeat_length; exact Hlt).
      apply nth_error_In, repeat_spec in Hfield. exact Hfield.
    + apply Nat.ltb_ge in Hlt.
      rewrite nth_error_app2 in Hfield by (rewrite repeat_length; exact Hlt).
      rewrite repeat_length, nth_error_map in Hfield. congruence.
  - injection Hkind as <-. apply nth_error_In, repeat_spec in Hfield.
    exact Hfield.
Qed.

Lemma simple_must_be_const_eval (vars : env) (s : simple) (c : const)
    (v : value) :
  simple_must_be_const s = Some c ->
  eval_simple vars s = Some v -> v = Value_const c.
Proof.
  destruct s as [x | sym | c']; simpl; try discriminate.
  intros [= ->] [= <-]. reflexivity.
Qed.

(* A constant of a kind of naked numbers is the constant of its
   number. *)
Lemma const_of_its_number (k : naked_number_kind) (c : const) (n : Z) :
  kind_of_const c = Naked_number k ->
  number_of_const c = Some n -> c = const_of_naked_number k n.
Proof.
  destruct c; simpl; try discriminate; intros [= <-] [= <-]; reflexivity.
Qed.

(* The one number of a set that has one. *)
Lemma get_singleton_member (numbers : list Z) (i j : Z) :
  Keyed.get_singleton Z.eqb numbers = Some i -> j ∈ numbers -> j = i.
Proof.
  intros Hsingleton Hin.
  pose proof (proj1 (keyed_get_singleton Z Z.eqb Z.eqb_eq i numbers)
                Hsingleton j)
    as Hamong.
  apply (keyed_among_in Z Z.eqb Z.eqb_eq) in Hin.
  rewrite Hamong in Hin. apply Z.eqb_eq in Hin. exact Hin.
Qed.

Lemma must_be_singleton_satisfied_all (statics : static_env) (vars : env) :
  (∀ (t : flambda_type) (v : value) (c : const),
    must_be_singleton t = Some (Some c) ->
    type_satisfied statics vars v t -> v = Value_const c)
  ∧ (∀ (head : head_of_kind_value) (v : value) (c : const),
      must_be_singleton (Type_value (Ok (No_alias head))) = Some (Some c) ->
      head_of_kind_value_satisfied statics vars v head -> v = Value_const c)
  ∧ (∀ (head : head_of_kind_value_non_null) (v : value) (c : const),
      must_be_singleton (non_null_value head) = Some (Some c) ->
      head_of_kind_value_non_null_satisfied statics vars v head ->
      v = Value_const c)
  ∧ (∀ blocks : row_like_for_blocks, True)
  ∧ (∀ closures : row_like_for_closures, True)
  ∧ (∀ entry : closures_entry, True)
  ∧ (∀ extension : env_extension, True).
Proof.
  apply types_induction; try (intros; exact I).
  - (* Type_value *)
    intros ty Hhead v c. destruct ty as [ | [head | s] | ].
    + discriminate.
    + autorewrite with type_satisfied. exact (Hhead v c).
    + simpl must_be_singleton. intros [= Hconst].
      autorewrite with type_satisfied. intros [Hs _].
      exact (simple_must_be_const_eval vars s c v Hconst Hs).
    + discriminate.
  - (* Type_naked_immediate *)
    intros ty v c. destruct ty as [ | [head | s] | ].
    + discriminate.
    + simpl must_be_singleton.
      destruct (naked_immediates head) as [numbers | ] eqn:Hnumbers;
        [ | discriminate ].
      destruct (Keyed.get_singleton Z.eqb numbers) as [i | ] eqn:Hsingleton;
        [ | discriminate ].
      intros [= <-]. autorewrite with type_satisfied.
      intros [j [-> [Hin _]]]. rewrite Hnumbers in Hin.
      rewrite (get_singleton_member numbers i j Hsingleton Hin). reflexivity.
    + simpl must_be_singleton. intros [= Hconst].
      autorewrite with type_satisfied. intros [Hs _].
      exact (simple_must_be_const_eval vars s c v Hconst Hs).
    + discriminate.
  - (* Type_naked_number *)
    intros k ty v c. destruct ty as [ | [numbers | s] | ].
    + discriminate.
    + simpl must_be_singleton.
      destruct (Keyed.get_singleton Z.eqb numbers) as [i | ] eqn:Hsingleton;
        [ | discriminate ].
      intros [= <-]. autorewrite with type_satisfied.
      intros [c' [n [-> [Hkind [Hnumber Hin]]]]].
      rewrite (const_of_its_number k c' n Hkind Hnumber),
        (get_singleton_member numbers i n Hsingleton Hin).
      reflexivity.
    + simpl must_be_singleton. intros [= Hconst].
      autorewrite with type_satisfied. intros [Hs _].
      exact (simple_must_be_const_eval vars s c v Hconst Hs).
    + discriminate.
  - intros ty v c. discriminate.
  - intros ty v c. discriminate.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null v c.
    destruct non_null as [ | head | ]; try discriminate.
    destruct is_null as [ | x]; [ | destruct head; discriminate ].
    rewrite head_of_kind_value_satisfied_eq.
    intros Hsingle [[_ []] | [_ [Hhead _]]].
    exact (Hnon_null v c Hsingle Hhead).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      _ _ v c.
    unfold non_null_value. simpl must_be_singleton.
    destruct blocks as [blocks | ]; [ | discriminate ].
    destruct (row_like_for_blocks_is_bottom blocks) eqn:Hbottom; simpl;
      [ | discriminate ].
    destruct immediates as [imm | ]; [ | discriminate ].
    destruct (must_be_singleton imm) as [[c' | ] | ] eqn:Hsingle;
      try discriminate.
    intros Hc. rewrite variant_satisfied.
    intros [[i [-> [_ [Himm _]]]] | [tag [fields [_ [_ [_ [Hrow _]]]]]]].
    + pose proof (Himmediates _ c' eq_refl Himm) as Hsame.
      injection Hsame as <-. injection Hc as <-. reflexivity.
    + destruct (row_like_for_blocks_is_bottom_satisfied statics vars tag
                  fields blocks Hbottom Hrow).
  - (* Mutable_block *)
    intros alloc_mode v c. discriminate.
  - (* Boxed_number *)
    intros k contents alloc_mode _ v c. discriminate.
  - (* Closures *)
    intros by_function_slot alloc_mode _ v c. discriminate.
  - (* String *)
    intros strs v c. discriminate.
Qed.

Theorem must_be_singleton_satisfied : must_be_singleton_satisfied_statement.
Proof.
  intros statics vars v t c.
  exact (proj1 (must_be_singleton_satisfied_all statics vars) t v c).
Qed.
