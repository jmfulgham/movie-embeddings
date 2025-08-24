import {openai, supabase} from './config.js';
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import fs from 'fs';

/*
  Challenge: Text Splitters, Embeddings, and Vector Databases!
    1. Use LangChain to split the content in movies.txt into smaller chunks.
    2. Use OpenAI's Embedding model to create an embedding for each chunk.
    3. Insert all text chunks and their corresponding embedding
       into a Supabase database table.
 */

const movies = fs.readFileSync('./movies.txt', {encoding: 'utf8', flag: 'r'});

/* Split movies.txt into text chunks.
Return LangChain's "output" – the array of Document objects. */
async function splitDocument(document) {
    //pull in the document to chunk

    if (!document) {
        console.error("Missing document for splitting, please try your request again");
        throw new Error("Missing document for splitting, please try your request again");
    }

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20
    });
    return splitter.createDocuments(document)
}

/* Create an embedding from each text chunk.
Store all embeddings and corresponding text in Supabase. */
async function createAndStoreEmbeddings() {
    const chunkData = await splitDocument([movies]);

    //for each chunk page content, create an embedding and store it
    try {
        const chunkedEmbeddings = await Promise.all(chunkData.map(async (chunk) => {
            const embeddingResponse = await openai.embeddings.create({
                model: "text-embedding-ada-002",
                input: chunk.pageContent,
            });

            if (!chunkedEmbeddings.ok) {
                throw new Error("Unable to create embeddings")
            }

            const embedding = embeddingResponse.data[0].embedding;
            return {
                content: chunk.pageContent,
                embedding
            }
        }))
        //      // Insert content and embedding into Supabase
        const {error} =await supabase.from('movies').insert(chunkedEmbeddings);
        if (error) {
            throw new Error("Unable to update database");
        }
    } catch (e) {
        console.error("Err: ", e.message)
    }
}


createAndStoreEmbeddings()
